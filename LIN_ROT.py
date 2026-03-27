"""
АД ADM174
Параметры машины:
  Un = 570 В (лин.), f = 50 Гц, p = 2, N = 28 стержней
  J = 2.875 кг*м^2, Mc_ном = 1096 Н*м
  Rs = 29.1 мОм, Rr = 20.17 мОм (приведённые к 3-фазам)
  Lls = 0.544 мГн, Llr = 0.476 мГн, Lm_dq = 17.228 мГн

Физические особенности АДМ174:
  • s_ном = 1.3% -> 2sf = 1.26 Гц (период биений ~796 мс)
  • J большой -> domega/omega < 0.01% -> примерно только одна гармоника 2sf в спектре
  • Im ~ 371 А = Is_rms * sqrt(3) ~ 210 * 1.732 (нормально для N-фазной системы)

Решатель: LSODA, max_step = 2e-4 с (~4 с на 3 с симуляции в норме)
Размер ОДУ: 2*28 + 2 = 58
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve as la_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter1d
import time as timer


# ПАРАМЕТРЫ
# Конфигурация симуляции
CONN_MODE    = 'star_grounded'  # 'star_grounded' | 'star_isolated'
LOCKED_ROTOR = False            # True = заторможенный ротор (theta=0, omega=0)

# Конфигурации обрывов для прогона
# Каждый элемент: (метка, список оборванных стержней)
FAULT_CONFIGS = [
    ("Норма",              []),
    ("1 стержень [0]",     [0]),
    ("2 стержня [0,1]",    [0, 1]),
    ("3 стержня [0,1,2]",  [0, 1, 2]),
    ("4 стержня [0..3]",   [0, 1, 2, 3]),
    ("2 несмежных [0,14]", [0, 14]),
]
R_FAULT = 1e6   # Ом - сопротивление оборванного стержня

# Время симуляции
T_END        = 3.0   # с - полный прогон (пуск + ХХ + нагрузка)
T_END_LR     = 0.5   # с - заторможенный ротор

# Нагрузка: Mc(t)
T_LOAD_ON    = 1.5   # с - момент включения нагрузки
MC_NOMINAL   = 1096.0  # Н*м

# Температурные профили [T_нач, ..., T_кон] C
# Один элемент = постоянная температура
TS_PROFILE   = [20]   # статор
TR_PROFILE   = [20]   # ротор

# Параметры машины ADM174
N_BARS   = 44            # число стержней ротора
P_POLES  = 2             # число пар полюсов

RS_20    = 0.0291        # Ом - сопротивление статора при 20 C
RR_20    = 0.02017       # Ом - сопротивление ротора при 20 C (приведённое)
ALPHA_R  = 0.004         # 1/C - температурный коэф. сопротивления

LLS      = 0.544e-3      # Гн - индуктивность рассеяния статора
LLR      = 0.476e-3      # Гн - индуктивность рассеяния ротора (приведённая)
LM_DQ    = 17.228e-3     # Гн - взаимоиндуктивность в dq-координатах
LM_3PH   = (2.0/3.0) * LM_DQ   # Гн - взаимоиндуктивность в 3-фазной схеме
LM       = LM_3PH * 2.0 / N_BARS   # Гн - на пару фаз в N-фазной системе

# Полные индуктивности фазы
LS       = LLS + LM
LR       = LLR + LM

J        = 2.875         # кг*м^2 - момент инерции
UN_LINE  = 570.0         # В - линейное напряжение
F_NET    = 50.0          # Гц - частота сети

# Производные константы
UN_PHASE = UN_LINE / np.sqrt(3)
U_AMP    = UN_PHASE * np.sqrt(2)
OMEGA1   = 2.0 * np.pi * F_NET
N_SYNC   = 60.0 * F_NET / P_POLES   # об/мин

# Решатель
ODE_METHOD  = 'LSODA'
ODE_MAXSTEP = 2e-4   # с
ODE_RTOL    = 1e-6
ODE_ATOL    = 1e-8


# ПРЕДВЫЧИСЛЕННЫЕ КОНСТАНТЫ (пересчитываются при изменении N_BARS)

_phi  = 2.0 * np.pi / N_BARS
_idx  = np.arange(N_BARS)

# Corr[i,j] = cos((j-i)*phi), диагональ = 0  (для Ls0, Lr0)
_Corr = np.cos((_idx[:, None] - _idx[None, :]) * _phi)
np.fill_diagonal(_Corr, 0.0)

# (i+j)*phi - угловая сетка для Cosr (зависит от theta)
_ij_phi = (_idx[:, None] + _idx[None, :]) * _phi

# Смещения фаз N-фазного напряжения (блок 2)
_phase_offsets = _idx * _phi

# Буфер для матрицы L
_L_buf  = np.empty((2 * N_BARS, 2 * N_BARS))

# Буфер для системы с множителем Лагранжа (star_isolated)
_L_iso  = np.empty((2 * N_BARS + 1, 2 * N_BARS + 1))
_cv     = np.zeros(2 * N_BARS)   # вектор ограничения sumIs = 0
_cv[:N_BARS] = 1.0

# Размер вектора состояния: 2N потокосцеплений + omega + y
_STATE = 2 * N_BARS + 2

# Цвета для графиков (норма + до 5 конфигураций обрывов)
_COLORS = ['#1f77b4', '#4daf4a', '#ff7f00', '#e41a1c', '#984ea3', '#a65628']


# ТЕМПЕРАТУРНЫЕ ИНТЕРПОЛЯТОРЫ

_Ts_func = [lambda t: 20.0]
_Tr_func = [lambda t: 20.0]


def _build_temp_interp(T_profile, t_sim_end):
    """Линейный интерполятор температуры по времени."""
    T = np.asarray(T_profile, dtype=float)
    if len(T) == 1:
        v = T[0]
        return lambda t: v
    nodes = np.linspace(0.0, t_sim_end, len(T))
    return lambda t: float(np.interp(t, nodes, T))


# МАТРИЦА ИНДУКТИВНОСТЕЙ 2Nx2N

def _build_L(gamma_e, out=None):
    """
    Собирает матрицу L(theta) размером 2Nx2N.

    Блочная структура:
      [ Ls0      Lsr(theta)^T ]
      [ Lsr(theta)   Lr0      ]

    Ls0[i,j] = Lm*Corr[i,j] + E_ij*Ls   (не зависит от theta)
    Lr0[i,j] = Lm*Corr[i,j] + E_ij*Lr   (не зависит от theta)
    Lrs[i,j] = Lm*cos(theta − (i+j)*phi)      (зависит от theta)
    """
    if out is None:
        out = np.empty((2 * N_BARS, 2 * N_BARS))

    Ls0 = LM * _Corr + np.diag(np.full(N_BARS, LS))
    Lr0 = LM * _Corr + np.diag(np.full(N_BARS, LR))
    Lrs = LM * np.cos(gamma_e - _ij_phi)

    out[:N_BARS, :N_BARS] = Ls0
    out[:N_BARS,  N_BARS:] = Lrs.T
    out[N_BARS:, :N_BARS] = Lrs
    out[N_BARS:,  N_BARS:] = Lr0
    return out


# НАПРЯЖЕНИЯ: 3-фазные -> N-фазные

def _voltages_n(t):
    """
    N-фазный вектор напряжений статора.
    Алгоритм: вычисляем амплитуду и фазу из 3-фазных,
    затем проецируем на N равноудалённых фаз.
    """
    wt = OMEGA1 * t
    Ua = U_AMP * np.sin(wt)
    Ub = U_AMP * np.sin(wt - 2.0 * np.pi / 3.0)
    Uc = U_AMP * np.sin(wt + 2.0 * np.pi / 3.0)
    dbc   = Ub - Uc
    Uabs  = np.sqrt(Ua * Ua + dbc * dbc / 3.0)
    alpha = np.arctan2(Ua, dbc / np.sqrt(3.0))
    return Uabs * np.cos(alpha + _phase_offsets)


def _voltages_full(t):
    """Полный вектор 2N: N статорных напряжений + N нулей (ротор КЗ)."""
    u = np.zeros(2 * N_BARS)
    u[:N_BARS] = _voltages_n(t)
    return u


def _voltages_3ph(t):
    """Три фазных напряжения для постобработки."""
    wt = OMEGA1 * t
    return (U_AMP * np.sin(wt),
            U_AMP * np.sin(wt - 2.0 * np.pi / 3.0),
            U_AMP * np.sin(wt + 2.0 * np.pi / 3.0))


# МОМЕНТ И ОГИБАЮЩАЯ

def _torque(Is, Ir, gamma_e):
    """
    Электромагнитный момент (формула 8 статьи):
      Te = −zp * Is^T * (Lm * sin_matrix) * Ir * 3/N
    Коэффициент 3/N обеспечивает паспортный момент
    независимо от размерности системы.
    """
    m = np.sin(gamma_e - _ij_phi)
    return -P_POLES * (Is @ (LM * m) @ Ir) * 3.0 / N_BARS


def _envelope(Is):
    """
    Огибающая тока статора (формула 10 статьи):
      Im = sqrt(sumIs^2 * 3/N)

    Связь с 3-фазным RMS: Im = Is_rms_3ph * sqrt3 ~ 210 * 1.732 ~ 363 А.
    Это не погрешность - Im есть модуль пространственного вектора тока
    в N-фазной системе, масштабированный к 3-фазной.
    """
    return np.sqrt(np.dot(Is, Is) * 3.0 / N_BARS)


def _envelope_smooth(t, Im, window_s=0.02):
    """
    Сглаженная огибающая (скользящее среднее с окном window_s секунд).
    Убирает пульсацию 100 Гц (двойная несущая), оставляет медленные биения.
    Используется для диагностического анализа: Im_smooth показывает пульсацию 2sf.
    """
    if len(t) < 4:
        return Im.copy()
    dt = (t[-1] - t[0]) / (len(t) - 1)
    n_win = max(1, int(window_s / dt))
    return uniform_filter1d(Im, size=n_win, mode='nearest')


# СХЕМА СОЕДИНЕНИЯ СТАТОРА: star_isolated

def _flux_to_currents_isolated(psi_full, gamma_e):
    """
    Восстановление токов при изолированной нейтрали.
    Ограничение: sumIs = 0 добавляется методом множителя Лагранжа.
    Расширенная система (2N+1) x (2N+1):
      [ L   c ] [ I ]   [ psi ]
      [ c^T 0 ] [ λ ] = [ 0 ]
    """
    _build_L(gamma_e, _L_buf)
    _L_iso[:2*N_BARS, :2*N_BARS] = _L_buf
    _L_iso[:2*N_BARS,  2*N_BARS] = _cv
    _L_iso[ 2*N_BARS, :2*N_BARS] = _cv
    _L_iso[ 2*N_BARS,  2*N_BARS] = 0.0
    rhs = np.zeros(2 * N_BARS + 1)
    rhs[:2*N_BARS] = psi_full
    return la_solve(_L_iso, rhs)[:2*N_BARS]


def _neutral_voltage(psi_full, i_full, u_src, Rs, Rr_vec):
    """Напряжение нейтрали для star_isolated."""
    R_vec = np.concatenate([np.full(N_BARS, Rs), Rr_vec])
    return np.mean((u_src - R_vec * i_full)[:N_BARS])


# ФУНКЦИИ ПАРАМЕТРОВ С УЧЁТОМ ТЕМПЕРАТУРЫ И ОБРЫВОВ

def _Rs_of_t(t):
    return RS_20 * (1.0 + ALPHA_R * (_Ts_func[0](t) - 20.0))


def _Rr_vec_of_t(t, faulty_bars, r_fault):
    """Вектор сопротивлений N стержней: учёт температуры и обрывов."""
    Rr = RR_20 * (1.0 + ALPHA_R * (_Tr_func[0](t) - 20.0))
    Rv = np.full(N_BARS, Rr)
    for bi in faulty_bars:
        Rv[bi] = r_fault
    return Rv


# ПРАВАЯ ЧАСТЬ ОДУ

_call_count = [0]
_wall_start = [0.0]
_last_wall  = [0.0]


def _make_rhs(faulty_bars, r_fault, locked_rotor, conn_mode, mc_func):
    """
    Фабрика правой части ОДУ.
    Замыкает конфигурацию: обрывы, нагрузку, схему соединения.
    Возвращает функцию ode(t, y) -> dy.
    """
    def ode(t, y):
        psi   = y[:2*N_BARS]
        omega = y[2*N_BARS]
        gamma = y[2*N_BARS + 1]

        if locked_rotor:
            omega = 0.0
            gamma = 0.0

        Rs     = _Rs_of_t(t)
        Rr_vec = _Rr_vec_of_t(t, faulty_bars, r_fault)
        R_vec  = np.concatenate([np.full(N_BARS, Rs), Rr_vec])

        if conn_mode == 'star_isolated':
            i_full = _flux_to_currents_isolated(psi, gamma)
        else:
            _build_L(gamma, _L_buf)
            i_full = la_solve(_L_buf, psi, check_finite=False)

        Is = i_full[:N_BARS]
        Ir = i_full[N_BARS:]

        u = _voltages_full(t)
        if conn_mode == 'star_isolated':
            u[:N_BARS] -= _neutral_voltage(psi, i_full, u, Rs, Rr_vec)

        dpsi = u - R_vec * i_full
        Te   = _torque(Is, Ir, gamma)

        if locked_rotor:
            domega, dgamma = 0.0, 0.0
        else:
            domega = (Te - mc_func(t)) / J
            dgamma = P_POLES * omega

        # Прогресс-вывод
        _call_count[0] += 1
        now = timer.time()
        if now - _last_wall[0] > 5.0:
            n_cur = omega * 30.0 / np.pi
            el    = now - _wall_start[0]
            print(f"    t={t*1000:8.2f} мс  n={n_cur:7.1f} об/мин  "
                  f"Im={_envelope(Is):6.1f} А  "
                  f"calls={_call_count[0]:>9d}  wall={el:.0f}с", flush=True)
            _last_wall[0] = now

        dy = np.empty(_STATE)
        dy[:2*N_BARS]    = dpsi
        dy[2*N_BARS]     = domega
        dy[2*N_BARS + 1] = dgamma
        return dy

    return ode


# ПОСТОБРАБОТКА:

def _postprocess(sol, faulty_bars, locked_rotor=False):
    """
    Полная постобработка решения.

    Возвращает словарь с:
      t, omega, n_rpm, slip       - механика
      iA, iB, iC, Is_rms          - 3-фазные токи статора
      Is_all (NxNt)               - все N статорных токов
      Ir_all (NxNt)               - все N токов стержней
      Me, Im                       - момент и огибающая
      Im_smooth                    - сглаженная огибающая (для диагностики)
      is0, i_neutral               - нулевая составляющая
      uA, uB, uC, Ps, Pmech        - напряжения и мощности
      Rs, Rr                       - сопротивления (температурный профиль)
      pq                           - разложение Акаги
    """
    t       = sol.t
    omega   = sol.y[2*N_BARS]
    gamma   = sol.y[2*N_BARS + 1]
    n_rpm   = omega * 30.0 / np.pi
    slip    = 1.0 - n_rpm / N_SYNC
    Nt      = len(t)

    iA = np.zeros(Nt); iB = np.zeros(Nt); iC = np.zeros(Nt)
    Me = np.zeros(Nt); Im = np.zeros(Nt)
    is0 = np.zeros(Nt); i_neut = np.zeros(Nt)
    Rs_a = np.zeros(Nt); Rr_a = np.zeros(Nt)
    uA = np.zeros(Nt);  uB = np.zeros(Nt);  uC = np.zeros(Nt)
    Ps = np.zeros(Nt)
    Is_all = np.zeros((N_BARS, Nt))
    Ir_all = np.zeros((N_BARS, Nt))

    INV_S3 = 1.0 / np.sqrt(3)
    S3     = np.sqrt(3)
    step   = N_BARS // 3   # индексы для извлечения 3-фазных токов

    print(f"  Постобработка ({Nt} точек)...", flush=True)
    t_pp = timer.time()

    for k in range(Nt):
        psi_k = sol.y[:2*N_BARS, k]
        ge_k  = gamma[k]

        if CONN_MODE == 'star_isolated':
            i_full = _flux_to_currents_isolated(psi_k, ge_k)
        else:
            _build_L(ge_k, _L_buf)
            i_full = la_solve(_L_buf, psi_k, check_finite=False)

        Is_k = i_full[:N_BARS]
        Ir_k = i_full[N_BARS:]

        Is_all[:, k] = Is_k
        Ir_all[:, k] = Ir_k

        # 3-фазные токи: Is[0]=iA, Is[N/3]=iB, Is[2N/3]=iC
        iA[k] = Is_k[0]; iB[k] = Is_k[step]; iC[k] = Is_k[2*step]

        Me[k] = _torque(Is_k, Ir_k, ge_k)
        Im[k] = _envelope(Is_k)

        is0[k]   = INV_S3 * (iA[k] + iB[k] + iC[k])
        i_neut[k] = S3 * is0[k]

        Rs_a[k] = _Rs_of_t(t[k])
        healthy  = [i for i in range(N_BARS) if i not in faulty_bars]
        Rr_a[k]  = RR_20 if not healthy else np.mean(
            [RR_20 * (1 + ALPHA_R * (_Tr_func[0](t[k]) - 20)) for _ in healthy])

        uA[k], uB[k], uC[k] = _voltages_3ph(t[k])
        Ps[k] = uA[k]*iA[k] + uB[k]*iB[k] + uC[k]*iC[k]

        if (k + 1) % 30000 == 0:
            print(f"    {k+1}/{Nt} ({(k+1)/Nt*100:.0f}%)", flush=True)

    print(f"  Постобработка за {timer.time()-t_pp:.1f} с")

    Is_rms  = np.sqrt((iA**2 + iB**2 + iC**2) / 3.0)
    Pmech   = Me * omega
    Im_smooth = _envelope_smooth(t, Im, window_s=0.02)

    pq = _pq_decompose(iA, iB, iC, uA, uB, uC)

    return dict(
        t=t, omega=omega, n_rpm=n_rpm, slip=slip,
        iA=iA, iB=iB, iC=iC, Is_rms=Is_rms,
        Is_all=Is_all, Ir_all=Ir_all,
        Me=Me, Im=Im, Im_smooth=Im_smooth,
        is0=is0, i_neutral=i_neut,
        uA=uA, uB=uB, uC=uC, Ps=Ps, Pmech=Pmech,
        Rs=Rs_a, Rr=Rr_a, pq=pq,
        faulty_bars=list(faulty_bars),
    )


# PQ-РАЗЛОЖЕНИЕ АКАГИ

_S3_2 = np.sqrt(3) / 2.0


def _pq_decompose(iA, iB, iC, uA, uB, uC):
    """Мгновенное pq-разложение."""
    ia = (2/3) * (iA - 0.5*iB - 0.5*iC)
    ib = (2/3) * _S3_2 * (iB - iC)
    ua = (2/3) * (uA - 0.5*uB - 0.5*uC)
    ub = (2/3) * _S3_2 * (uB - uC)
    p  = ua*ia + ub*ib
    q  = ub*ia - ua*ib
    us2  = ua**2 + ub**2
    us2s = np.where(us2 > 1e-9, us2, 1.0)
    iap = ua/us2s*p;  ibp = ub/us2s*p
    iaq = ub/us2s*q;  ibq = -ua/us2s*q
    iAp = iap;  iBp = -0.5*iap + _S3_2*ibp;  iCp = -0.5*iap - _S3_2*ibp
    iAq = iaq;  iBq = -0.5*iaq + _S3_2*ibq;  iCq = -0.5*iaq - _S3_2*ibq
    Ip  = np.sqrt((iAp**2 + iBp**2 + iCp**2) / 3.0)
    Iq  = np.sqrt((iAq**2 + iBq**2 + iCq**2) / 3.0)
    It  = np.sqrt(Ip**2 + Iq**2)
    cosphi = np.where(It > 1e-9, Ip / np.where(It > 1e-9, It, 1.0), 0.0)
    return dict(ia=ia, ib=ib, ua=ua, ub=ub, p=p, q=q,
                iAp=iAp, iBp=iBp, iCp=iCp,
                iAq=iAq, iBq=iBq, iCq=iCq,
                Ip=Ip, Iq=Iq, cosphi=cosphi)


# ГАРМОНИЧЕСКИЙ АНАЛИЗ

def harmonic_analysis(t, x, f_fund, n_periods=4, n_harm=30):
    """
    Гармонический анализ сигнала x(t) на последних n_periods периодах.
    Возвращает: (t_окно, x_окно, таблица[k, f_k, amp_k], THD%).
    """
    T     = 1.0 / f_fund
    t_sta = t[-1] - n_periods * T
    if t_sta < t[0]:
        t_sta = t[0]
    m   = t >= t_sta
    tw  = t[m]; xw = x[m]; Nw = len(tw)
    if Nw < 4:
        return tw, xw, np.zeros((n_harm, 3)), 0.0
    dt    = (tw[-1] - tw[0]) / (Nw - 1)
    freqs = np.fft.rfftfreq(Nw, d=dt)
    X     = np.fft.rfft(xw) / Nw
    amps  = 2.0 * np.abs(X); amps[0] /= 2.0
    res   = np.zeros((n_harm, 3))
    for k in range(n_harm):
        idx = np.argmin(np.abs(freqs - k * f_fund))
        res[k] = [k, freqs[idx], amps[idx]]
    A1  = res[1, 2]
    thd = (np.sqrt(np.sum(res[2:, 2]**2)) / A1 * 100.0) if A1 > 1e-12 else 0.0
    return tw, xw, res, thd


def envelope_spectrum(t, Im, t_start, smooth_w=0.02, f_max=10.0):
    """
    БПФ сглаженной огибающей для диагностического анализа.
    Возвращает (freqs, amps) в диапазоне [0, f_max] Гц.
    """
    m  = t >= t_start; tw = t[m]; Iw = Im[m]
    if len(tw) < 16:
        return np.zeros(2), np.zeros(2)
    dt  = (tw[-1] - tw[0]) / (len(tw) - 1)
    Ism = uniform_filter1d(Iw, size=max(1, int(smooth_w / dt)), mode='nearest')
    Iac = Ism - np.mean(Ism)
    X   = np.fft.rfft(Iac) / len(Iac)
    fr  = np.fft.rfftfreq(len(Iac), d=dt)
    am  = 2.0 * np.abs(X)
    m_f = fr <= f_max
    return fr[m_f], am[m_f]


# ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА ОДНОЙ СИМУЛЯЦИИ

def run_simulation(label, faulty_bars=None, r_fault=None,
                   locked_rotor=False, t_end_override=None,
                   Ts_prof=None, Tr_prof=None,
                   mc_override=None):
    """
    Запуск одной симуляции и постобработка.

    Параметры:
      label         - метка для вывода
      faulty_bars   - список номеров оборванных стержней ([] = норма)
      r_fault       - сопротивление обрыва, Ом (по умолч. R_FAULT)
      locked_rotor  - True = заторможенный ротор
      t_end_override - время симуляции (по умолч. T_END или T_END_LR)
      Ts_prof, Tr_prof - профили температур
      mc_override   - функция Mc(t), заменяет стандартную

    Возвращает словарь результатов или None при ошибке.
    """
    if faulty_bars is None:
        faulty_bars = []
    if r_fault is None:
        r_fault = R_FAULT

    sim_end = t_end_override if t_end_override is not None else (
        T_END_LR if locked_rotor else T_END)

    Ts_p = Ts_prof if Ts_prof is not None else TS_PROFILE
    Tr_p = Tr_prof if Tr_prof is not None else TR_PROFILE
    _Ts_func[0] = _build_temp_interp(Ts_p, sim_end)
    _Tr_func[0] = _build_temp_interp(Tr_p, sim_end)

    # Момент нагрузки
    if mc_override is not None:
        mc_func = mc_override
    else:
        mc_func = lambda t: 0.0 if t < T_LOAD_ON else MC_NOMINAL

    # Заголовок
    lr_tag = " [ЗАТОРМ. РОТОР]" if locked_rotor else ""
    f_tag  = f", обрыв {faulty_bars}" if faulty_bars else ", норма"
    print(f"\n{'='*70}")
    print(f"  АД ADM174 - {N_BARS}-фазная модель (Глазырин 2021){lr_tag}")
    print(f"  Схема: {CONN_MODE}{f_tag}, R_fault={r_fault:.0e} Ом")
    print(f"  Lm_3ph={LM_3PH*1e3:.4f} мГн, Lm={LM*1e6:.2f} мкГн, "
          f"Ls={LS*1e3:.4f} мГн, Lr={LR*1e3:.4f} мГн")
    print(f"  Ts={Ts_p} C, Tr={Tr_p} C, dim={_STATE}")
    print(f"{'='*70}")
    print(f"\n  Интегрирование (t_end={sim_end} с)...", flush=True)

    _call_count[0] = 0
    t0 = timer.time()
    _wall_start[0] = t0; _last_wall[0] = t0

    ode = _make_rhs(faulty_bars, r_fault, locked_rotor, CONN_MODE, mc_func)
    sol = solve_ivp(ode, [0, sim_end], np.zeros(_STATE),
                    method=ODE_METHOD, max_step=ODE_MAXSTEP,
                    rtol=ODE_RTOL, atol=ODE_ATOL)

    elapsed = timer.time() - t0
    print(f"\n  Готово за {elapsed:.1f} с  ({len(sol.t)} точек, "
          f"{_call_count[0]} вызовов, OK={sol.success})")
    if not sol.success:
        print(f"  ОШИБКА: {sol.message}")
        return None

    res = _postprocess(sol, faulty_bars, locked_rotor)
    res['label'] = label

    # Вывод итогов
    _print_results(res, locked_rotor, Ts_p, Tr_p)
    return res


def _print_results(res, locked_rotor, Ts_p, Tr_p):
    """Вывод численных результатов в консоль."""
    print(f"\n  {'─'*60}")
    print(f"  РЕЗУЛЬТАТЫ  Ts {Ts_p[0]:.0f}->{Ts_p[-1]:.0f} C  "
          f"Tr {Tr_p[0]:.0f}->{Tr_p[-1]:.0f} C")
    print(f"  Rs {res['Rs'][0]*1e3:.3f}->{res['Rs'][-1]*1e3:.3f} мОм  "
          f"Rr {res['Rr'][0]*1e3:.4f}->{res['Rr'][-1]*1e3:.4f} мОм")
    print(f"  {'─'*60}")

    if locked_rotor:
        print(f"  > Заторм. ротор (t={res['t'][-1]:.2f} с уст.):"
              f"\n      Me={res['Me'][-1]:.2f} Н*м  "
              f"Im={res['Im'][-1]:.2f} А  Is={res['Is_rms'][-1]:.1f} А  "
              f"Ps={res['Ps'][-1]/1e3:.1f} кВт")
    else:
        ix_xx = np.searchsorted(res['t'], min(1.49, res['t'][-1] - 0.01))
        s_xx  = res['slip'][ix_xx]
        print(f"  > ХХ (t~1.49 с):"
              f"\n      n={res['n_rpm'][ix_xx]:.1f} об/мин  s={s_xx:.5f}"
              f"  2sf={2*s_xx*F_NET:.3f} Гц"
              f"\n      Me={res['Me'][ix_xx]:.2f} Н*м  "
              f"Im={res['Im'][ix_xx]:.2f} А  Is={res['Is_rms'][ix_xx]:.1f} А")
        print(f"  > Нагрузка (t={res['t'][-1]:.2f} с):"
              f"\n      n={res['n_rpm'][-1]:.1f} об/мин  s={res['slip'][-1]:.5f}"
              f"\n      Me={res['Me'][-1]:.2f} Н*м  "
              f"Im={res['Im'][-1]:.2f} А  Is={res['Is_rms'][-1]:.1f} А"
              f"\n      Ps={res['Ps'][-1]/1e3:.1f} кВт  "
              f"Pmech={res['Pmech'][-1]/1e3:.1f} кВт")
        pq = res['pq']
        print(f"  > Акаги (ХХ): "
              f"Ip={pq['Ip'][ix_xx]:.1f} А  Iq={pq['Iq'][ix_xx]:.1f} А  "
              f"cos_phi={pq['cosphi'][ix_xx]:.4f}")
        print(f"  > Пуск: Max|Me|={np.max(np.abs(res['Me'])):.0f} Н*м  "
              f"Max Im={np.max(res['Im']):.1f} А")


# ГРАФИКИ

plt.rcParams.update({
    'font.size': 9, 'figure.dpi': 150,
    'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 0.6,
})


def _vline(ax, t_ms=T_LOAD_ON * 1000):
    ax.axvline(t_ms, c='red', lw=0.6, ls='--', alpha=0.45)


def plot_full_run(res, fname='plot_full.png'):
    """
    Рис. L1 - полная картина одного прогона:
    токи статора, токи стержней, момент, скорость,
    потокосцепления, скольжение, мех. хар-ка,
    Im/Is_rms, мощности, нулевые составляющие, Rs/Rr, cos_phi.
    """
    t   = res['t'] * 1000
    fig = plt.figure(figsize=(17, 28))
    fig.suptitle(f'АД ADM174 - {N_BARS}-фазная модель'
                 f'{CONN_MODE}  {res["label"]}',
                 fontsize=11, fontweight='bold', y=0.995)
    gs  = GridSpec(7, 2, hspace=0.45, wspace=0.25,
                   left=0.07, right=0.97, top=0.96, bottom=0.03)
    cs3 = ['#d62728', '#1f77b4', '#2ca02c']

    def _ax(row, col): return fig.add_subplot(gs[row, col])

    ax = _ax(0, 0)
    ax.plot(t, res['iA'], c=cs3[0], label='$i_A$')
    ax.plot(t, res['iB'], c=cs3[1], label='$i_B$')
    ax.plot(t, res['iC'], c=cs3[2], label='$i_C$')
    ax.set_ylabel('А'); ax.set_title('Токи статора (3-фазные)'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(0, 1)
    for bi, c in zip([0, N_BARS//4, N_BARS//2], cs3):
        ax.plot(t, res['Ir_all'][bi], c=c, label=f'стержень {bi}')
    ax.set_ylabel('А'); ax.set_title('Токи стержней ротора (выборка)'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(1, 0)
    ax.plot(t, res['Me'], c='#9467bd')
    ax.axhline(MC_NOMINAL, c='red', ls=':', lw=0.5, alpha=0.5, label=f'Mc={MC_NOMINAL:.0f}')
    ax.set_ylabel('Н*м'); ax.set_title('Момент'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(1, 1)
    ax.plot(t, res['n_rpm'], c='#ff7f0e')
    ax.axhline(N_SYNC, c='gray', ls=':', lw=0.7, label=f'n_0={N_SYNC:.0f}')
    ax.set_ylabel('об/мин'); ax.set_title('Скорость'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(2, 0)
    for bi, c in zip(range(3), cs3):
        ax.plot(t, res['Is_all'][bi], c=c, label=f'psi_s{bi+1}')
    ax.set_ylabel('Вб'); ax.set_title('Потокосцепл. статора (psi по N-фазным токам)'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(2, 1)
    for bi, c in zip(range(3), cs3):
        ax.plot(t, res['Ir_all'][bi] * LR, c=c, ls='--', label=f'psi_r{bi+1}')
    ax.set_ylabel('Вб'); ax.set_title('Потокосцепл. ротора (Ir*Lr)'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(3, 0)
    ax.plot(t, res['slip'], c='#8c564b')
    ax.set_ylabel('s'); ax.set_title('Скольжение'); _vline(ax)

    ax = _ax(3, 1)
    ax.plot(res['n_rpm'], res['Me'], c='#9467bd', lw=0.15, alpha=0.6)
    ax.axhline(0, c='k', lw=0.3)
    ax.set_xlabel('n, об/мин'); ax.set_ylabel('Н*м'); ax.set_title('Механическая хар-ка')

    ax = _ax(4, 0)
    ax.plot(t, res['Im'], c='#17becf', label='Im (огибающая)')
    ax.plot(t, res['Im_smooth'], c='navy', lw=0.8, ls='--', label='Im_сгл')
    ax.set_ylabel('А'); ax.set_title('Огибающая тока статора  Im = sqrt(sumIs^2*3/N)'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(4, 1)
    ax.plot(t, res['Is_rms'], c='#17becf')
    ax.axhline(57.45,  c='green', ls=':', lw=0.7, label='I_0=57.45')
    ax.axhline(204.42, c='gray',  ls=':', lw=0.7, label='Iн=204.42')
    ax.set_ylabel('А'); ax.set_title('3-фазный RMS тока статора'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(5, 0)
    ax.plot(t, res['Ps']/1e3,    c='#e377c2', label='$P_{эл}$')
    ax.plot(t, res['Pmech']/1e3, c='#bcbd22', label='$P_{мех}$')
    ax.axhline(170, c='gray', ls=':', lw=0.7, label='$P_{2н}$=170')
    ax.set_ylabel('кВт'); ax.set_title('Мощности'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(5, 1)
    ax.plot(t, res['is0'],       c='#ff6600', label='$i_s^{(0)}$')
    ax.plot(t, res['i_neutral'], c='#9467bd', ls='--', lw=0.5, label='$i_n$')
    ax.set_ylabel('А'); ax.set_title(f'Нулевые составляющие ({CONN_MODE})'); ax.legend(fontsize=7); _vline(ax)

    ax = _ax(6, 0)
    ax.plot(t, res['Rs']*1e3, c='#d62728', label='$R_s$')
    ax.plot(t, res['Rr']*1e3, c='#1f77b4', label='$R_r$')
    ax.set_ylabel('мОм'); ax.set_title('Сопротивления (темп. зависимость)'); ax.legend(fontsize=7)

    ax = _ax(6, 1)
    ax.plot(t, res['pq']['cosphi'], c='#2ca02c')
    ax.set_ylabel('cos phi'); ax.set_title('Коэффициент мощности (мгновенный)'); _vline(ax)

    for ax in fig.get_axes():
        if ax.get_xlabel() == '':
            ax.set_xlabel('t, мс')

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fname}")


def plot_fault_comparison(all_results, fname='plot_faults.png'):
    """
    Рис. сравнения - момент, скорость, огибающая для всех конфигураций.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'АД ADM174: сравнение конфигураций обрывов ({CONN_MODE})',
                 fontsize=11, fontweight='bold')

    for (lbl, r), c in zip(all_results.items(), _COLORS):
        if r is None: continue
        t = r['t'] * 1000
        axes[0].plot(t, r['Me'],    c=c, lw=0.6, label=lbl, alpha=0.9)
        axes[1].plot(t, r['n_rpm'], c=c, lw=0.8, label=lbl)
        axes[2].plot(t, r['Im'],    c=c, lw=0.6, label=lbl, alpha=0.9)

    for ax in axes:
        _vline(ax); ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel('Н*м'); axes[0].set_title('Момент')
    axes[1].set_ylabel('об/мин'); axes[1].set_title('Скорость')
    axes[2].set_ylabel('Im, А'); axes[2].set_xlabel('t, мс')
    axes[2].set_title('Огибающая тока статора')

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fname}")


def plot_envelope_detail(all_results, fname='plot_envelope_detail.png'):
    """
    Детальный анализ огибающей:
    панель 1 - полная Im под нагрузкой (все конфигурации)
    панель 2 - сглаженная Im (биения 2sf видны)
    панель 3 - нормированная переменная составляющая (%)
    панель 4 - спектр (БПФ) с маркерами 2sf, 4sf
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 18))

    # Оценим диагностическую частоту по нормальному режиму
    r0 = next((v for v in all_results.values() if v is not None), None)
    if r0 is None:
        plt.close(); return
    m_ss = r0['t'] >= 2.0
    s_mean = np.mean(r0['slip'][m_ss])
    f_diag = 2.0 * s_mean * F_NET
    T_beat = 1.0 / f_diag if f_diag > 0 else 1.0

    fig.suptitle(f'АД ADM174: анализ огибающей тока статора\n'
                 f'2sf = {f_diag:.3f} Гц  (T_beat = {T_beat*1000:.0f} мс)',
                 fontsize=11, fontweight='bold')

    t_load = T_LOAD_ON
    for (lbl, r), c in zip(all_results.items(), _COLORS):
        if r is None: continue
        t = r['t']; Im = r['Im']
        m = t >= t_load; tm = t[m] - t_load

        # 1. Полная огибающая
        axes[0].plot(tm, Im[m], c=c, lw=0.6, label=lbl, alpha=0.9)

        # 2. Сглаженная
        Im_sm = _envelope_smooth(t[m], Im[m], window_s=0.02)
        axes[1].plot(tm, Im_sm, c=c, lw=0.8, label=lbl)

        # 3. Нормированная переменная составляющая
        Im_mean = np.mean(Im[t >= 2.0])
        Im_ac   = (Im_sm - Im_mean) / Im_mean * 100.0
        axes[2].plot(tm, Im_ac, c=c, lw=0.8, label=lbl)

        # 4. Спектр
        fr, am = envelope_spectrum(t[m] + t_load, Im[m], t_start=2.0, f_max=8.0)
        if len(fr) > 2:
            axes[3].plot(fr, am, c=c, lw=0.8, label=lbl, alpha=0.9)

    # Маркеры периодов биений
    t_end_plot = r0['t'][-1] - t_load
    for kb in range(1, int(t_end_plot * f_diag) + 2):
        tb = kb * T_beat
        if tb <= t_end_plot:
            axes[0].axvline(tb, c='gray', lw=0.5, ls='--', alpha=0.2)
            axes[1].axvline(tb, c='gray', lw=0.5, ls='--', alpha=0.2)
            axes[2].axvline(tb, c='gray', lw=0.5, ls='--', alpha=0.2)

    # Маркеры гармоник
    for kh in range(1, 4):
        fk = kh * f_diag
        if fk < 8.0:
            axes[3].axvline(fk, c='gray', lw=0.8, ls=':', alpha=0.6,
                            label=f'{kh}x2sf={fk:.2f}Гц' if kh == 1 else f'{kh}x2sf')

    axes[0].set_ylabel('Im, А'); axes[0].set_title('Мгновенная огибающая')
    axes[0].legend(fontsize=7, ncol=3)
    axes[1].set_ylabel('Im_сгл, А'); axes[1].set_title('Сглаженная огибающая (окно 20 мс)')
    axes[1].legend(fontsize=7, ncol=3)
    axes[2].set_ylabel('%'); axes[2].set_title('Нормированная переменная составляющая')
    axes[2].axhline(0, c='k', lw=0.4)
    axes[2].legend(fontsize=7, ncol=3)
    axes[3].set_ylabel('А'); axes[3].set_xlabel('f, Гц')
    axes[3].set_title('Спектр огибающей (БПФ, t>2 с)')
    axes[3].legend(fontsize=7, ncol=3); axes[3].set_xlim(0, 8)

    for ax in axes[:3]:
        ax.set_xlabel('t, с'); ax.set_xlim(0, t_end_plot)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fname}")


def plot_bar_currents(r_norm, r_fault, fname='plot_bars.png'):
    """
    Токи стержней: норма vs обрыв (первые 5 стержней, режим нагрузки).
    """
    t_center = r_fault['t'][-1] - 1.5 * (1.0 / F_NET)
    n_show   = 5
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), sharex=True)
    fig.suptitle(f'Токи стержней ротора: норма vs {r_fault["label"]}\n'
                 f'Режим нагрузки, 3 периода', fontsize=11, fontweight='bold')
    T2 = 2.0 * (1.0 / F_NET)

    for pi in range(n_show):
        ax = axes[pi]
        for r, c, lbl in [(r_norm, '#1f77b4', 'норма'),
                          (r_fault, '#d62728', r_fault['label'])]:
            m = (r['t'] >= t_center - T2) & (r['t'] <= t_center + T2)
            ls = '--' if pi in r['faulty_bars'] else '-'
            lbl_full = lbl + (' [ОБРЫВ]' if pi in r['faulty_bars'] else '')
            ax.plot((r['t'][m] - t_center) * 1000, r['Ir_all'][pi][m],
                    c=c, lw=0.7, ls=ls, label=lbl_full, alpha=0.9)
        ax.set_ylabel(f'Ir[{pi}], А'); ax.legend(fontsize=7)
    axes[-1].set_xlabel('t, мс (от центра окна)')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fname}")


def plot_harmonic_analysis(res, mode_label='', fname='plot_harmonics.png'):
    """
    Гармонический анализ токов статора (ХХ и нагрузка).
    """
    t = res['t']
    iA = res['iA']; iB = res['iB']; iC = res['iC']
    i_n = res['i_neutral']

    # ХХ
    m_xx = (t >= 1.42) & (t <= 1.50)
    _, _, hA_xx, thdA_xx = harmonic_analysis(t[m_xx], iA[m_xx], F_NET, 4)
    _, _, hB_xx, thdB_xx = harmonic_analysis(t[m_xx], iB[m_xx], F_NET, 4)
    _, _, hC_xx, thdC_xx = harmonic_analysis(t[m_xx], iC[m_xx], F_NET, 4)
    _, _, h0_xx, thd0_xx = harmonic_analysis(t[m_xx], i_n[m_xx], F_NET, 4)

    # Нагрузка
    _, _, hA_ld, thdA_ld = harmonic_analysis(t, iA, F_NET, 4)
    _, _, hB_ld, thdB_ld = harmonic_analysis(t, iB, F_NET, 4)
    _, _, hC_ld, thdC_ld = harmonic_analysis(t, iC, F_NET, 4)
    _, _, h0_ld, thd0_ld = harmonic_analysis(t, i_n, F_NET, 4)

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f'Гармонический анализ токов статора - {res["label"]} {mode_label}',
                 fontsize=11, fontweight='bold')
    cs3 = ['#d62728', '#1f77b4', '#2ca02c']

    for col, (h_xx, h_ld, thd_xx, thd_ld, lbl, c) in enumerate([
        (hA_xx, hA_ld, thdA_xx, thdA_ld, 'iA', cs3[0]),
        (hB_xx, hB_ld, thdB_xx, thdB_ld, 'iB', cs3[1]),
    ]):
        axes[col, 0].bar(h_xx[1:, 0], h_xx[1:, 2], color=c, width=0.7, alpha=0.8)
        axes[col, 0].set_title(f'{lbl} - ХХ (THD={thd_xx:.2f}%)')
        axes[col, 0].set_xlabel('Номер гармоники'); axes[col, 0].set_ylabel('А')
        axes[col, 1].bar(h_ld[1:, 0], h_ld[1:, 2], color=c, width=0.7, alpha=0.8)
        axes[col, 1].set_title(f'{lbl} - нагрузка (THD={thd_ld:.2f}%)')
        axes[col, 1].set_xlabel('Номер гармоники'); axes[col, 1].set_ylabel('А')

    axes[2, 0].bar(hC_xx[1:, 0], hC_xx[1:, 2], color=cs3[2], width=0.7, alpha=0.8)
    axes[2, 0].set_title(f'iC - ХХ (THD={thdC_xx:.2f}%)')
    axes[2, 0].set_xlabel('Номер гармоники'); axes[2, 0].set_ylabel('А')
    axes[2, 1].bar(hC_ld[1:, 0], hC_ld[1:, 2], color=cs3[2], width=0.7, alpha=0.8)
    axes[2, 1].set_title(f'iC - нагрузка (THD={thdC_ld:.2f}%)')
    axes[2, 1].set_xlabel('Номер гармоники'); axes[2, 1].set_ylabel('А')

    axes[3, 0].bar(h0_xx[1:, 0], h0_xx[1:, 2], color='#ff6600', width=0.7, alpha=0.8)
    axes[3, 0].set_title(f'i_n - ХХ (THD={thd0_xx:.2f}%)')
    axes[3, 0].set_xlabel('Номер гармоники'); axes[3, 0].set_ylabel('А')
    axes[3, 1].bar(h0_ld[1:, 0], h0_ld[1:, 2], color='#ff6600', width=0.7, alpha=0.8)
    axes[3, 1].set_title(f'i_n - нагрузка (THD={thd0_ld:.2f}%)')
    axes[3, 1].set_xlabel('Номер гармоники'); axes[3, 1].set_ylabel('А')

    for ax in axes.flat:
        ax.set_xlim(0, 20)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fname}")


def plot_pq(res, fname='plot_pq.png'):
    """pq-разложение Акаги - активная/реактивная составляющие."""
    t = res['t'] * 1000
    pq = res['pq']
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'pq-разложение Акаги - {res["label"]}', fontsize=11, fontweight='bold')
    cs3 = ['#d62728', '#1f77b4', '#2ca02c']

    axes[0,0].plot(t, res['iA'], c=cs3[0], lw=0.5, label='$i_A$')
    axes[0,0].plot(t, pq['iAp'], c='#ff7f0e', ls='--', label='$i_{Ap}$')
    axes[0,0].set_title('$i_A$ и активная составляющая'); _vline(axes[0,0]); axes[0,0].legend(fontsize=7)

    axes[0,1].plot(t, res['iA'], c=cs3[0], lw=0.5, label='$i_A$')
    axes[0,1].plot(t, pq['iAq'], c='#9467bd', ls='--', label='$i_{Aq}$')
    axes[0,1].set_title('$i_A$ и реактивная составляющая'); _vline(axes[0,1]); axes[0,1].legend(fontsize=7)

    axes[1,0].plot(t, pq['p']/1e3, c='#e377c2', lw=0.5)
    axes[1,0].set_title('Мгновенная мощность p'); _vline(axes[1,0]); axes[1,0].set_ylabel('кВт')

    axes[1,1].plot(t, pq['q']/1e3, c='#17becf', lw=0.5)
    axes[1,1].set_title('Мгновенная мощность q'); _vline(axes[1,1]); axes[1,1].set_ylabel('квар')

    axes[2,0].plot(t, pq['Ip'], c='#ff7f0e', lw=0.5, label='$I_p$ (акт.)')
    axes[2,0].plot(t, res['Is_rms'], c='gray', lw=0.3, ls='--', label='$I_s$ полн.')
    axes[2,0].set_title('Активный ток'); _vline(axes[2,0]); axes[2,0].legend(fontsize=7)

    axes[2,1].plot(t, pq['cosphi'], c='#2ca02c', lw=0.5)
    axes[2,1].set_title('Коэффициент мощности'); _vline(axes[2,1]); axes[2,1].set_ylabel('cos phi')

    for ax in axes.flat:
        ax.set_xlabel('t, мс')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fname}")


def plot_startup(res, fname='plot_startup.png'):
    """Детали пуска 0–200 мс."""
    m = res['t'] <= 0.2; t = res['t'][m] * 1000
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f'Пуск 0–200 мс - {res["label"]}', fontsize=11, fontweight='bold')
    cs3 = ['#d62728', '#1f77b4', '#2ca02c']
    for iph, c, lbl in zip([res['iA'], res['iB'], res['iC']], cs3, ['iA','iB','iC']):
        axes[0,0].plot(t, iph[m], c=c, label=f'${lbl}$')
    axes[0,0].legend(fontsize=7); axes[0,0].set_title('Токи статора')
    axes[0,1].plot(t, res['Me'][m], c='#9467bd'); axes[0,1].set_title('Момент'); axes[0,1].set_ylabel('Н*м')
    axes[1,0].plot(t, res['n_rpm'][m], c='#ff7f0e'); axes[1,0].set_title('Скорость'); axes[1,0].set_ylabel('об/мин')
    axes[1,1].plot(t, res['Im'][m], c='#17becf', label='Im')
    axes[1,1].plot(t, res['Is_rms'][m], c='gray', ls='--', lw=0.5, label='Is_rms')
    axes[1,1].legend(fontsize=7); axes[1,1].set_title('Ток статора')
    for ax in axes.flat:
        ax.set_xlabel('t, мс')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fname}")



# СВОДНАЯ ТАБЛИЦА

def print_summary(all_results):
    """Сводная таблица результатов для всех конфигураций."""
    print(f"\n{'='*80}")
    print("  СВОДНАЯ ТАБЛИЦА")
    print(f"{'='*80}")
    print(f"\n  {'Конфигурация':<24}  {'n_ХХ':>8}  {'Im_ХХ':>7}  "
          f"{'n_нагр':>8}  {'s_нагр':>7}  {'Im_нагр':>8}  {'Me_нагр':>9}  {'Is_нагр':>9}")
    print("  " + "─" * 86)
    for lbl, r in all_results.items():
        if r is None:
            print(f"  {lbl:<24}  - (ошибка)")
            continue
        ix = np.searchsorted(r['t'], 1.49)
        s_ld = r['slip'][-1]
        print(f"  {lbl:<24}  "
              f"{r['n_rpm'][ix]:>8.1f}  {r['Im'][ix]:>7.2f}  "
              f"{r['n_rpm'][-1]:>8.1f}  {s_ld:>7.4f}  "
              f"{r['Im'][-1]:>8.2f}  {r['Me'][-1]:>9.1f}  "
              f"{r['Is_rms'][-1]:>9.1f}")


# ТОЧКА ВХОДА

if __name__ == '__main__':

    print(f"\n{'='*70}")
    print(f"  АД ADM174 - N-фазная модель (Глазырин)")
    print(f"{'='*70}")
    print(f"  N_BARS  = {N_BARS}  |  dim ОДУ = {_STATE}  |  CONN_MODE = {CONN_MODE}")
    print(f"  Lm_3ph  = {LM_3PH*1e3:.4f} мГн  ->  Lm = {LM*1e6:.4f} мкГн")
    print(f"  Rs_20   = {RS_20*1e3:.3f} мОм  |  Rr_20 = {RR_20*1e3:.4f} мОм")
    print(f"  J = {J} кг*м^2  |  Un = {UN_LINE} В  |  f = {F_NET} Гц  |  p = {P_POLES}")
    print(f"  Mc_ном = {MC_NOMINAL} Н*м  (с t = {T_LOAD_ON} с)")

    # Прогоны для всех конфигураций обрывов
    all_results    = {}   # полные прогоны 0–T_END
    all_results_lr = {}   # заторможенный ротор

    for lbl, faults in FAULT_CONFIGS:
        all_results[lbl] = run_simulation(
            label=lbl, faulty_bars=faults)
        all_results_lr[lbl] = run_simulation(
            label=f"{lbl} [ЗР]", faulty_bars=faults,
            locked_rotor=True)

    # Сводная таблица
    print_summary(all_results)

    # Гармонический анализ нормального режима
    r_norm = all_results.get("Норма")
    if r_norm is not None:
        print(f"\n{'='*70}")
        print(f"  ГАРМОНИЧЕСКИЙ АНАЛИЗ (норма, {CONN_MODE})")
        print(f"{'='*70}")
        for mode, t_sig, x_sig, lbl in [
            ('ХХ',      r_norm['t'][(r_norm['t']>=1.42)&(r_norm['t']<=1.50)],
                        r_norm['iA'][(r_norm['t']>=1.42)&(r_norm['t']<=1.50)], 'iA'),
            ('Нагр.',   r_norm['t'], r_norm['iA'], 'iA'),
        ]:
            _, _, h, thd = harmonic_analysis(t_sig, x_sig, F_NET, 4)
            print(f"  {mode} {lbl}: фунд.={h[1,2]:.2f} А, THD={thd:.2f}%")

    # Графики
    print(f"\n{'='*70}")
    print("  ПОСТРОЕНИЕ ГРАФИКОВ")
    print(f"{'='*70}")

    if r_norm is not None:
        plot_full_run(r_norm, 'plot_L1_full_norm.png')
        plot_startup(r_norm,  'plot_L2_startup.png')
        plot_pq(r_norm,       'plot_L3_pq.png')
        plot_harmonic_analysis(r_norm, fname='plot_L4_harmonics.png')

    plot_fault_comparison(all_results,    'plot_G1_fault_comparison.png')
    plot_envelope_detail(all_results,     'plot_G2_envelope_analysis.png')

    if r_norm is not None:
        r_f = all_results.get("4 стержня [0..3]") or all_results.get("3 стержня [0,1,2]")
        if r_f is not None:
            plot_bar_currents(r_norm, r_f, 'plot_G3_bar_currents.png')

    print(f"\n{'='*70}")
    print("  ГОТОВО.")
    print(f"{'='*70}")
