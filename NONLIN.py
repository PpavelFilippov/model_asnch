"""
Исправления v5:
  A) Добавлен учёт нулевых составляющих токов и напряжений
     по методике Дроздовского-Дуды.
  B) Два режима подключения статора:
       'star_grounded'  -- звезда с заземлённой нейтралью
       'star_isolated'  -- звезда без нейтрали (iA+iB+iC = 0)
  C) Расчёт и визуализация:
       i_s^(0) = (iA+iB+iC)/sqrt3  -- нулевая составляющая тока
       u_s^(0) = (uA+uB+uC)/sqrt3  -- нулевая составляющая напряжения
       u_n = u_s^(0)/sqrt3          -- напряжение нейтрали
  D) Гармонический анализ нулевых составляющих (3-я гармоника от насыщения).

Кривая: Psi_m(Im) = a * arctan(b * Im)

"""

import numpy as np
from numpy.linalg import solve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time as timer


# Вспомогательные функции вывода

def _header(title, ch='=', width=72):
    """Заголовок секции: линия -> заголовок -> линия."""
    line = ch * width
    print(f"\n{line}\n  {title}\n{line}")

def _subheader(title, ch='-', width=60):
    """Подзаголовок секции."""
    line = ch * width
    print(f"\n  {line}\n  {title}\n  {line}")

def _section(title):
    """Лёгкий маркер раздела внутри секции."""
    print(f"\n  === {title} ===")

def _table_header(*cols, widths=None):
    """Шапка таблицы с выравниванием."""
    if widths is None:
        widths = [30] + [10] * (len(cols) - 1)
    fmt = "    " + "  ".join(f"{{:>{w}s}}" if i else f"{{:{w}s}}"
                              for i, w in enumerate(widths))
    print(fmt.format(*cols))

def _fig_saved(name):
    """Подтверждение сохранения рисунка."""
    print(f"  + Рис. {name} сохранён")

def _harm_table_header_4col():
    """Стандартная шапка таблицы гармоник (iA, iB, iC, i_n)."""
    print(f"  {'Гарм':>5}  {'f, Гц':>8}  {'iA':>8}  {'iB':>8}  {'iC':>8}  {'i_n':>9}")

def _harm_table_row(k, freq, aA, aB, aC, aN):
    """Одна строка таблицы гармоник."""
    print(f"  {k:5d}  {freq:8.1f}  {aA:8.2f}  {aB:8.2f}  {aC:8.2f}  {aN:9.3f}")

def _thd_line(thd_A, thd_B, thd_C, prefix=""):
    """Строка THD для трёх фаз."""
    print(f"  {prefix}THD iA = {thd_A:.2f}%,  THD iB = {thd_B:.2f}%,  THD iC = {thd_C:.2f}%")

def _compare_header():
    """Шапка таблицы сравнения (насыщ vs линейн)."""
    _table_header('Параметр', 'Насыщ', 'Линейн', 'Разность')


CONN_MODE = 'star_grounded'

SATURATION_ENABLED = True


p_poles = 2
Rs_20 = 0.0291     # Ом, сопротивление фазы статора при 20C
Rr_20 = 0.02017    # Ом, сопротивление фазы ротора при 20C
alpha_R = 0.004     # 1/C, температурный коэффициент

# Профили температуры: задаём массивами, размножаем на сетку
# равномерно по времени моделирования.
#   Ts_profile -- температура статора, C
#   Tr_profile -- температура ротора, C
# Если массив из одного элемента -- температура постоянна.

Ts_profile = [20]
Tr_profile = [20]

# Начальные (при 20°C) -- для обратной совместимости
RsA = Rs_20;  RsB = Rs_20;  RsC = Rs_20
Rra = Rr_20;  Rrb = Rr_20;  Rrc = Rr_20
R_phases = np.array([RsA, RsB, RsC, Rra, Rrb, Rrc])


def _build_temp_interp(T_profile, t_sim_end):
    """
    Строит интерполятор температуры по времени.
    T_profile -- list/array температур, равномерно распределённых
                 от t=0 до t=t_sim_end.
    Возвращает функцию T(t).
    """
    T_arr = np.asarray(T_profile, dtype=float)
    n = len(T_arr)
    if n == 1:
        val = T_arr[0]
        return lambda t: val
    t_nodes = np.linspace(0.0, t_sim_end, n)
    return lambda t: np.interp(t, t_nodes, T_arr)


_Ts_func = [lambda t: 20.0]   # placeholder, обновляется в run_simulation
_Tr_func = [lambda t: 20.0]


def R_phases_of_t(t):
    """
    Вектор сопротивлений [RsA,RsB,RsC, Rra,Rrb,Rrc] при температуре T(t).
    R(T) = R_20 * (1 + alpha * (T - 20))
    Нагрев равномерный по фазам.
    """
    Ts = _Ts_func[0](t)
    Tr = _Tr_func[0](t)
    Rs = Rs_20 * (1.0 + alpha_R * (Ts - 20.0))
    Rr = Rr_20 * (1.0 + alpha_R * (Tr - 20.0))
    return np.array([Rs, Rs, Rs, Rr, Rr, Rr])

Lls = 0.544e-3
Llr = 0.476e-3
Lm_dq_nom = 17.228e-3
Lm_nom = (2.0 / 3.0) * Lm_dq_nom


Lls0 = Lls

J = 2.875
Un_line = 570.0
Un_phase = Un_line / np.sqrt(3)
U_amp = Un_phase * np.sqrt(2)
f = 50.0
omega1 = 2 * np.pi * f
n_sync = 60 * f / p_poles
sigma_nom = 1 - Lm_nom**2 / ((Lls+Lm_nom) * (Llr+Lm_nom))
t_end = 3.0


K_sat = 1.263
Lm_dq_0 = Lm_dq_nom * K_sat
Lm_0 = (2.0 / 3.0) * Lm_dq_0
Im_nom_dq = 57.45

_b_mag = 0.017019
_a_mag = Lm_dq_0 / _b_mag
_Psi_max_dq = _a_mag * np.pi / 2


def psi_m_dq(Im_dq):
    """Psi_m_dq(Im_dq) -- потокосцепление взаимоиндукции в dq."""
    return _a_mag * np.arctan(_b_mag * Im_dq)


def Lm_dq_of_Im(Im_dq):
    """Lm_dq = Psi_m_dq / Im_dq.  При Im->0: Lm_dq(0) = Lm_dq_nom x Ksat."""
    bI = _b_mag * Im_dq
    if bI < 1e-12:
        return Lm_dq_0
    return Lm_dq_0 * np.arctan(bI) / bI


def Lm_3ph_of_Im(Im_dq):
    """Lm для трёхфазной модели = (2/3) * Lm_dq."""
    return (2.0 / 3.0) * Lm_dq_of_Im(Im_dq)


_SR_ANGLES = np.array([
    [ 0.0,            2*np.pi/3,   -2*np.pi/3],
    [-2*np.pi/3,      0.0,          2*np.pi/3],
    [ 2*np.pi/3,     -2*np.pi/3,   0.0       ]
])

_SQRT3_2 = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)
_INV_SQRT3 = 1.0 / np.sqrt(3)
_TWO_THIRDS = 2.0 / 3.0
_SQRT2_OVER3 = np.sqrt(2) / 3.0


def _build_L(theta, Lm, out=None):
    """Сборка L 6x6 in-place."""
    Ls = Lls + Lm
    Lr = Llr + Lm
    hLm = Lm * 0.5
    cos_sr = Lm * np.cos(theta + _SR_ANGLES)

    if out is None:
        out = np.empty((6, 6))

    out[0,0]=Ls;    out[0,1]=-hLm;  out[0,2]=-hLm
    out[1,0]=-hLm;  out[1,1]=Ls;    out[1,2]=-hLm
    out[2,0]=-hLm;  out[2,1]=-hLm;  out[2,2]=Ls

    out[0,3]=cos_sr[0,0]; out[0,4]=cos_sr[0,1]; out[0,5]=cos_sr[0,2]
    out[1,3]=cos_sr[1,0]; out[1,4]=cos_sr[1,1]; out[1,5]=cos_sr[1,2]
    out[2,3]=cos_sr[2,0]; out[2,4]=cos_sr[2,1]; out[2,5]=cos_sr[2,2]

    out[3,0]=cos_sr[0,0]; out[3,1]=cos_sr[1,0]; out[3,2]=cos_sr[2,0]
    out[4,0]=cos_sr[0,1]; out[4,1]=cos_sr[1,1]; out[4,2]=cos_sr[2,1]
    out[5,0]=cos_sr[0,2]; out[5,1]=cos_sr[1,2]; out[5,2]=cos_sr[2,2]

    out[3,3]=Lr;    out[3,4]=-hLm;  out[3,5]=-hLm
    out[4,3]=-hLm;  out[4,4]=Lr;    out[4,5]=-hLm
    out[5,3]=-hLm;  out[5,4]=-hLm;  out[5,5]=Lr

    return out


def _compute_Im_dq(i6, cos_sr):
    """
    Модуль тока намагничивания Im_dq (действующее значение, RMS).

    1. Привести ir к статору: ir'_k = Sum_j cos(theta+phi_kj) * ir_j
    2. im_k = is_k + ir'_k   (трёхфазные координаты)
    3. Преобразование Кларк -> действующее значение вектора:
         коэффициент sqrt2/3  =  (2/3) * (1/sqrt2)
       где 2/3 -- amplitude-invariant Кларк,
           1/sqrt2 -- перевод амплитуды в действующее.
    4. Im_dq = sqrt(im_d^2 + im_q^2)

    ЗАМЕЧАНИЕ: нулевая составляющая тока намагничивания НЕ участвует
    в формировании основного магнитного потока (она не создаёт
    вращающегося поля). Поэтому Im_dq вычисляется только из alpha-beta
    компонент, как и раньше.
    """
    ir = i6[3:6]
    ir_s0 = cos_sr[0,0]*ir[0] + cos_sr[0,1]*ir[1] + cos_sr[0,2]*ir[2]
    ir_s1 = cos_sr[1,0]*ir[0] + cos_sr[1,1]*ir[1] + cos_sr[1,2]*ir[2]
    ir_s2 = cos_sr[2,0]*ir[0] + cos_sr[2,1]*ir[1] + cos_sr[2,2]*ir[2]

    im0 = i6[0] + ir_s0
    im1 = i6[1] + ir_s1
    im2 = i6[2] + ir_s2


    im_alpha = _SQRT2_OVER3 * (im0 - 0.5*im1 - 0.5*im2)
    im_beta  = _SQRT2_OVER3 * _SQRT3_2 * (im1 - im2)

    return np.sqrt(im_alpha*im_alpha + im_beta*im_beta)


def zero_seq_current(i6):
    """
    Нулевая составляющая тока статора:
      i_s^(0) = (iA + iB + iC) / sqrt3

    Это совпадает с формулой из статьи Дроздовского:
      i_s^(0) = (1/sqrt3)(iU + iV + iW)

    Ток нейтрали: i_n = iA + iB + iC = sqrt3 * i_s^(0)

    Для схемы 'star_isolated' эта величина == 0 (ограничение).
    Для схемы 'star_grounded' она != 0 из-за насыщения (3-я гармоника).
    """
    return _INV_SQRT3 * (i6[0] + i6[1] + i6[2])


def zero_seq_voltage(uA, uB, uC):
    """
    Нулевая составляющая напряжений статора:
      u_s^(0) = (uA + uB + uC) / sqrt3

    Для симметричного трёхфазного источника: u_s^(0) = 0.
    Напряжение нейтрали: u_n = u_s^(0) / sqrt3
    """
    return _INV_SQRT3 * (uA + uB + uC)


def neutral_voltage_from_zero_seq(u_s0):
    """u_n = u_s^(0) / sqrt3, т.е. u_s^(0) = sqrt3 * u_n"""
    return u_s0 / _SQRT3


_L_buf = np.empty((6, 6))


def flux_to_currents_sat(psi6, theta, Lm_prev):
    """Итерационное решение Psi = L(theta, Lm(Im)) * i."""
    if not SATURATION_ENABLED:
        Lm_fixed = Lm_nom
        _build_L(theta, Lm_fixed, _L_buf)
        i6 = solve(_L_buf, psi6)
        return i6, Lm_fixed

    cos_sr = np.cos(theta + _SR_ANGLES)
    Lm = Lm_prev

    for k in range(30):
        _build_L(theta, Lm, _L_buf)
        i6 = solve(_L_buf, psi6)

        Im = _compute_Im_dq(i6, cos_sr)
        Lm_new = Lm_3ph_of_Im(Im)

        err = abs(Lm_new - Lm)
        if err < 1e-9:
            return i6, Lm_new

        if err < 1e-6:
            alpha = 0.7
        elif err < 1e-4:
            alpha = 0.5
        else:
            alpha = 0.3
        Lm = alpha * Lm_new + (1.0 - alpha) * Lm

    return i6, Lm


_L_buf_7 = np.empty((7, 7))


def flux_to_currents_sat_isolated(psi6, theta, Lm_prev):
    """
    Решение Psi = L(theta, Lm(Im)) * i с ограничением iA + iB + iC = 0.

    Расширяем систему 6x6 до 7x7 с множителем Лагранжа (= u_n):
      [L  | c ] [i ] = [Psi]
      [c^T | 0 ] [lambda_ ]   [0]
    где c = [1, 1, 1, 0, 0, 0]^T -- ограничение на сумму токов статора.

    lambda_ физически связано с напряжением нейтрали.
    """
    cos_sr = np.cos(theta + _SR_ANGLES)
    constraint = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    if not SATURATION_ENABLED:
        Lm_fixed = Lm_nom
        _build_L(theta, Lm_fixed, _L_buf)
        _L_buf_7[:6, :6] = _L_buf
        _L_buf_7[:6, 6] = constraint
        _L_buf_7[6, :6] = constraint
        _L_buf_7[6, 6] = 0.0
        rhs = np.zeros(7)
        rhs[:6] = psi6
        rhs[6] = 0.0
        sol7 = solve(_L_buf_7, rhs)
        return sol7[:6], Lm_fixed

    Lm = Lm_prev

    for k in range(30):
        _build_L(theta, Lm, _L_buf)


        _L_buf_7[:6, :6] = _L_buf
        _L_buf_7[:6, 6] = constraint
        _L_buf_7[6, :6] = constraint
        _L_buf_7[6, 6] = 0.0

        rhs = np.zeros(7)
        rhs[:6] = psi6
        rhs[6] = 0.0

        sol7 = solve(_L_buf_7, rhs)
        i6 = sol7[:6]


        Im = _compute_Im_dq(i6, cos_sr)
        Lm_new = Lm_3ph_of_Im(Im)

        err = abs(Lm_new - Lm)
        if err < 1e-9:
            return i6, Lm_new

        if err < 1e-6:
            alpha = 0.7
        elif err < 1e-4:
            alpha = 0.5
        else:
            alpha = 0.3
        Lm = alpha * Lm_new + (1.0 - alpha) * Lm

    return i6, Lm


def electromagnetic_torque(i6, theta, Lm, Im_dq):
    """
    Электромагнитный момент через производную коэнергии по theta.

    Коэнергия W'(i, theta):
      W' = W'_leak + W'_mag(im, theta)

    W'_leak = (1/2)*Lls*(is^2) + (1/2)*Llr*(ir^2) -- не зависит от theta.

    W'_mag зависит от theta через взаимное расположение обмоток.
    Для определения dW'/dtheta при i=const нужно учесть, что при
    фиксированных фазных токах и повороте ротора на dtheta:
      - Lm(Im) меняется, т.к. Im зависит от theta через приведение ir->is
      - Lsr(theta) меняет cos -> sin

    Полная производная коэнергии:
      Me = dW'/dtheta |_{i=const}

    Для магнитной коэнергии в трёхфазных координатах:
      W'_mag = integral_0^{im} psi_m_3ph(x) dx
    где psi_m_3ph -- потокосцепление одной фазы от тока намагничивания.

    Однако эту коэнергию сложно выразить аналитически в 3-фазных
    координатах при произвольном theta.
    """
    sin_sr = -Lm * np.sin(theta + _SR_ANGLES)
    is3 = i6[0:3]
    ir3 = i6[3:6]
    return p_poles * (is3 @ sin_sr @ ir3)


LOCKED_ROTOR = False

_last_Lm = [Lm_0]
_call_count = [0]
_wall_start = [0.0]
_last_wall = [0.0]


def Mc_func(t, omega_mech):
    return 0.0 if t < 1.5 else 1096.0


def supply_voltages_phase(t):
    """Фазные напряжения источника (относительно нейтрали источника)."""
    uA = U_amp * np.cos(omega1 * t)
    uB = U_amp * np.cos(omega1 * t - 2*np.pi/3)
    uC = U_amp * np.cos(omega1 * t + 2*np.pi/3)
    return uA, uB, uC


def supply_voltages(t):
    """
    Вектор напряжений 6x1 в зависимости от схемы соединения.

    star_grounded: uA, uB, uC подаются напрямую -- ток нейтрали != 0.
    star_isolated: uA, uB, uC (напряжение нейтрали вычитается в ОДУ).
    """
    uA, uB, uC = supply_voltages_phase(t)

    return np.array([uA, uB, uC, 0.0, 0.0, 0.0])


def compute_neutral_voltage(psi6, i6, u6_source, theta, Lm, R_ph=None):
    """
    Вычисление напряжения нейтрали u_n для схемы star_isolated.

    Из уравнения напряжений для каждой фазы:
      u_k - u_n = R_s * i_k + dPsi_k/dt

    Суммируя по трём фазам и используя условие d(iA+iB+iC)/dt = 0:
      (uA+uB+uC) - 3*u_n = Rs*(iA+iB+iC) + d(PsiA+PsiB+PsiC)/dt

    Для star_isolated: iA+iB+iC = 0 (и его производная тоже = 0).
    Сумма потокосцеплений статора:
      PsiA+PsiB+PsiC = (Lls - Lm/2 - Lm/2)*(iA+iB+iC) + ...
    Из условия iA+iB+iC = 0 упрощается.

    На практике проще: u_n = (1/3)*[(uA+uB+uC) - Rs*(iA+iB+iC)
                               - d(PsiA+PsiB+PsiC)/dt]
    Мы берём u_n из текущего шага как:
      u_n = (1/3)*(uA+uB+uC) - (Rs/3)*(iA+iB+iC) - (1/3)*d(PsiA+PsiB+PsiC)/dt

    Поскольку iA+iB+iC = 0 для isolated, и (uA+uB+uC) = 0 для симметричного
    источника, u_n определяется из (1/3)*d(PsiA+PsiB+PsiC)/dt = -u_n.

    Фактически u_n уже содержится в разности:
      dpsi_dt_unconstrained = u_source - R*i6
    а затем: u_n = (1/3)*sum(dpsi_dt_unconstrained[:3])
    """
    if R_ph is None:
        R_ph = R_phases
    dpsi_unconstrained = u6_source - R_ph * i6

    sum_dpsi_s = dpsi_unconstrained[0] + dpsi_unconstrained[1] + dpsi_unconstrained[2]


    u_n = sum_dpsi_s / 3.0
    return u_n


def ode_system(t, y):
    psi6 = y[0:6]
    omega_mech = y[6]
    theta_elec = y[7]

    if LOCKED_ROTOR:
        omega_mech = 0.0
        theta_elec = 0.0

    if CONN_MODE == 'star_isolated':
        i6, Lm_c = flux_to_currents_sat_isolated(psi6, theta_elec, _last_Lm[0])
    else:
        i6, Lm_c = flux_to_currents_sat(psi6, theta_elec, _last_Lm[0])
    _last_Lm[0] = Lm_c

    u6 = supply_voltages(t)
    R_ph = R_phases_of_t(t)

    if CONN_MODE == 'star_isolated':

        u_n = compute_neutral_voltage(psi6, i6, u6, theta_elec, Lm_c, R_ph)
        u6[0] -= u_n
        u6[1] -= u_n
        u6[2] -= u_n

    dpsi_dt = u6 - R_ph * i6

    cos_sr = np.cos(theta_elec + _SR_ANGLES)
    Im = _compute_Im_dq(i6, cos_sr)
    Me_val = electromagnetic_torque(i6, theta_elec, Lm_c, Im)

    if LOCKED_ROTOR:
        domega_dt = 0.0
        dtheta_dt = 0.0
    else:
        Mc = Mc_func(t, omega_mech)
        domega_dt = (Me_val - Mc) / J
        dtheta_dt = p_poles * omega_mech

    _call_count[0] += 1
    wt = timer.time()
    if wt - _last_wall[0] > 5.0:
        n_cur = omega_mech * 30 / np.pi
        el = wt - _wall_start[0]
        i0 = zero_seq_current(i6)
        print(f"    t={t*1000:8.2f} мс  n={n_cur:7.1f} об/мин  "
              f"Lm={Lm_c*1e3:.3f} мГн  Im={Im:.1f} А  "
              f"i0={i0:.2f} А  "
              f"calls={_call_count[0]:>9d}  wall={el:.0f}с", flush=True)
        _last_wall[0] = wt

    return np.concatenate([dpsi_dt, [domega_dt, dtheta_dt]])


def run_simulation(saturation_on, label, locked_rotor=False, t_end_override=None,
                   Ts_prof=None, Tr_prof=None):
    """Запуск интегрирования и постобработка. Возвращает словарь результатов."""
    global SATURATION_ENABLED, LOCKED_ROTOR
    SATURATION_ENABLED = saturation_on
    LOCKED_ROTOR = locked_rotor
    sim_t_end = t_end_override if t_end_override is not None else t_end

    # Температурные профили
    Ts_p = Ts_prof if Ts_prof is not None else Ts_profile
    Tr_p = Tr_prof if Tr_prof is not None else Tr_profile
    _Ts_func[0] = _build_temp_interp(Ts_p, sim_t_end)
    _Tr_func[0] = _build_temp_interp(Tr_p, sim_t_end)

    Ts_start = Ts_p[0]; Ts_end = Ts_p[-1]
    Tr_start = Tr_p[0]; Tr_end = Tr_p[-1]

    mode_str = "С НАСЫЩЕНИЕМ" if saturation_on else "БЕЗ НАСЫЩЕНИЯ (Lm=const)"
    lr_str = "  [ЗАТОРМОЖЕННЫЙ РОТОР]" if locked_rotor else ""
    _header(f"АД ADM174 -- {mode_str}{lr_str}\n"
            f"  Схема соединения: {CONN_MODE}\n"
            f"  Температура: Ts = {Ts_p} °C,  Tr = {Tr_p} °C")

    if saturation_on:
        print(f"  Kнас = {K_sat}")
        print(f"  Lm_dq_nom = {Lm_dq_nom*1e3:.3f} мГн  (при Im_nom = {Im_nom_dq} А)")
        print(f"  Lm_dq(0)  = {Lm_dq_0*1e3:.3f} мГн  (ненасыщенное)")
        print(f"  a = {_a_mag:.4f} Вб,  b = {_b_mag:.6f} 1/А")
        print(f"  Psi_m_max = {_Psi_max_dq:.3f} Вб")
    else:
        print(f"  Lm_3ph = {Lm_nom*1e3:.3f} мГн = const")
        print(f"  Lm_dq  = {Lm_dq_nom*1e3:.3f} мГн = const")

    print(f"\n  Интегрирование (t_end = {sim_t_end} с)...", flush=True)
    t0 = timer.time()
    y0 = np.zeros(8)
    _last_Lm[0] = Lm_0 if saturation_on else Lm_nom
    _call_count[0] = 0
    _wall_start[0] = t0
    _last_wall[0] = t0

    sol = solve_ivp(
        ode_system, [0, sim_t_end], y0,
        method='Radau', max_step=5e-5,
        rtol=1e-6, atol=1e-8,
    )

    elapsed = timer.time() - t0
    print(f"\n  Готово за {elapsed:.1f} с  ({len(sol.t)} точек, "
          f"{_call_count[0]} вызовов, успех: {sol.success})")

    t_arr = sol.t
    omega_m = sol.y[6]
    theta_e = sol.y[7]
    n_rpm_arr = omega_m * 30 / np.pi
    slip_arr = 1 - n_rpm_arr / n_sync

    Nt = len(t_arr)
    iA_ = np.zeros(Nt); iB_ = np.zeros(Nt); iC_ = np.zeros(Nt)
    ia_ = np.zeros(Nt); ib_ = np.zeros(Nt); ic_ = np.zeros(Nt)
    Me_ = np.zeros(Nt); Lm_a = np.zeros(Nt); Im_a = np.zeros(Nt)
    is0_ = np.zeros(Nt)
    i_neut = np.zeros(Nt)
    us0_ = np.zeros(Nt)
    u_neut = np.zeros(Nt)
    Rs_arr = np.zeros(Nt)
    Rr_arr = np.zeros(Nt)

    print(f"  Постобработка ({Nt} точек)...", flush=True)
    t0p = timer.time()
    Lm_c = Lm_0 if saturation_on else Lm_nom
    for k in range(Nt):
        if CONN_MODE == 'star_isolated':
            i6, Lm_c = flux_to_currents_sat_isolated(sol.y[:6, k], theta_e[k], Lm_c)
        else:
            i6, Lm_c = flux_to_currents_sat(sol.y[:6, k], theta_e[k], Lm_c)

        iA_[k]=i6[0]; iB_[k]=i6[1]; iC_[k]=i6[2]
        ia_[k]=i6[3]; ib_[k]=i6[4]; ic_[k]=i6[5]

        cos_sr = np.cos(theta_e[k] + _SR_ANGLES)
        Im_a[k] = _compute_Im_dq(i6, cos_sr)
        Me_[k] = electromagnetic_torque(i6, theta_e[k], Lm_c, Im_a[k])
        Lm_a[k] = Lm_c

        is0_[k] = zero_seq_current(i6)
        i_neut[k] = _SQRT3 * is0_[k]

        R_ph_k = R_phases_of_t(t_arr[k])
        Rs_arr[k] = R_ph_k[0]
        Rr_arr[k] = R_ph_k[3]

        uA_k, uB_k, uC_k = supply_voltages_phase(t_arr[k])

        if CONN_MODE == 'star_grounded':
            us0_[k] = _INV_SQRT3 * (uA_k + uB_k + uC_k)
            u_neut[k] = 0.0
        elif CONN_MODE == 'star_isolated':
            u6 = np.array([uA_k, uB_k, uC_k, 0.0, 0.0, 0.0])
            u_n = compute_neutral_voltage(sol.y[:6, k], i6, u6, theta_e[k], Lm_c, R_ph_k)
            u_neut[k] = u_n
            us0_[k] = _SQRT3 * u_n

        if (k+1) % 20000 == 0:
            print(f"    {k+1}/{Nt} ({(k+1)/Nt*100:.0f}%)", flush=True)

    print(f"  Постобработка за {timer.time()-t0p:.1f} с")

    Is_rms_ = np.sqrt((iA_**2 + iB_**2 + iC_**2) / 3)
    Ps_ = np.zeros(Nt)
    for k in range(Nt):
        u = supply_voltages(t_arr[k])
        Ps_[k] = u[0]*iA_[k] + u[1]*iB_[k] + u[2]*iC_[k]
    Pmech_ = Me_ * omega_m

    idx_pre = np.searchsorted(t_arr, min(1.49, t_arr[-1] - 0.01))
    _subheader(f"РЕЗУЛЬТАТЫ ({label}, {CONN_MODE})\n"
               f"  Температура: Ts {Ts_start:.0f}->{Ts_end:.0f} °C,  "
               f"Tr {Tr_start:.0f}->{Tr_end:.0f} °C\n"
               f"  Rs: {Rs_arr[0]*1e3:.3f} -> {Rs_arr[-1]*1e3:.3f} мОм,  "
               f"Rr: {Rr_arr[0]*1e3:.3f} -> {Rr_arr[-1]*1e3:.3f} мОм")

    if locked_rotor:
        print(f"\n  > ЗАТОРМОЖЕННЫЙ РОТОР (установившийся, t = {t_arr[-1]:.2f} с):"
              f"\n      n = 0 об/мин,  s = 1.0"
              f"\n      Me (уст.) = {Me_[-1]:.2f} Н*м"
              f"\n      Is (уст.) = {Is_rms_[-1]:.1f} А"
              f"\n      Ps (уст.) = {Ps_[-1]/1e3:.1f} кВт"
              f"\n      Lm_3ph = {Lm_a[-1]*1e3:.3f} мГн"
              f"\n      Im_dq = {Im_a[-1]:.1f} А"
              f"\n      i_n (уст.) = {i_neut[-1]:.3f} А")
        print(f"\n  > Переходный процесс:"
              f"\n      Max|Me| = {np.max(np.abs(Me_)):.0f} Н*м"
              f"\n      Max|iA| = {np.max(np.abs(iA_)):.0f} А"
              f"\n      Max|Is| = {np.max(Is_rms_):.0f} А"
              f"\n      Lm min  = {np.min(Lm_a)*1e3:.3f} мГн"
              f"\n      Im max  = {np.max(Im_a):.0f} А"
              f"\n      Max|i_n| = {np.max(np.abs(i_neut)):.3f} А")
    else:
        print(f"\n  > ХХ (t ~= 1.49 с):"
              f"\n      n = {n_rpm_arr[idx_pre]:.1f} об/мин,  s = {slip_arr[idx_pre]:.6f}"
              f"\n      Me = {Me_[idx_pre]:.2f} Н*м"
              f"\n      Is = {Is_rms_[idx_pre]:.1f} А  (паспорт Io = 57.45 А)"
              f"\n      Lm_3ph = {Lm_a[idx_pre]*1e3:.3f} мГн"
              f"\n      Im_dq = {Im_a[idx_pre]:.1f} А"
              f"\n      i_n = {i_neut[idx_pre]:.3f} А")
        print(f"\n  > Нагрузка 1096 Н*м (t = {t_arr[-1]:.2f} с):"
              f"\n      n = {n_rpm_arr[-1]:.1f} об/мин,  s = {slip_arr[-1]:.6f}"
              f"\n      Me = {Me_[-1]:.2f} Н*м"
              f"\n      Is = {Is_rms_[-1]:.1f} А  (паспорт In = 204.42 А)"
              f"\n      Ps = {Ps_[-1]/1e3:.1f} кВт,  Pmech = {Pmech_[-1]/1e3:.1f} кВт"
              f"\n      i_n = {i_neut[-1]:.3f} А")
        print(f"\n  > Пуск:"
              f"\n      Max|Me| = {np.max(np.abs(Me_)):.0f} Н*м"
              f"\n      Max|iA| = {np.max(np.abs(iA_)):.0f} А"
              f"\n      Lm min  = {np.min(Lm_a)*1e3:.3f} мГн"
              f"\n      Im max  = {np.max(Im_a):.0f} А"
              f"\n      Max|i_n| = {np.max(np.abs(i_neut)):.3f} А")

    return dict(
        t=t_arr, sol=sol,
        PsiA=sol.y[0], PsiB=sol.y[1], PsiC=sol.y[2],
        Psi_a=sol.y[3], Psi_b=sol.y[4], Psi_c=sol.y[5],
        omega=omega_m, theta=theta_e,
        n_rpm=n_rpm_arr, slip=slip_arr,
        iA=iA_, iB=iB_, iC=iC_,
        ia=ia_, ib=ib_, ic=ic_,
        Me=Me_, Lm=Lm_a, Im=Im_a,
        is0=is0_, i_neutral=i_neut,
        us0=us0_, u_neutral=u_neut,
        Is_rms=Is_rms_, Ps=Ps_, Pmech=Pmech_,
        Rs=Rs_arr, Rr=Rr_arr,
        label=label,
    )


print("\n  Контрольные точки кривой намагничивания (power-invariant dq):")
for Im in [0, 30, 57.45, 100, 200, 500, 1000, 2000]:
    Ld = Lm_dq_of_Im(Im)
    psi = psi_m_dq(Im)
    r = Ld / Lm_dq_nom * 100
    print(f"    Im={Im:7.1f} А:  Psi_m={psi:.4f} Вб  "
          f"Lm_dq={Ld*1e3:.3f} мГн ({r:.1f}% ном)  "
          f"Lm_3ph={Ld*2/3*1e3:.3f} мГн")

res_sat = run_simulation(saturation_on=True, label="С насыщением")
res_lin = run_simulation(saturation_on=False, label="Без насыщения (Lm=const)")


# ЗАТОРМОЖЕННЫЙ РОТОР (Locked Rotor) -- s = 1, omega = 0


t_end_lr = 0.5
res_lr_sat = run_simulation(saturation_on=True, label="Заторм. ротор (насыщ)",
                            locked_rotor=True, t_end_override=t_end_lr)
res_lr_lin = run_simulation(saturation_on=False, label="Заторм. ротор (линейн)",
                            locked_rotor=True, t_end_override=t_end_lr)

LOCKED_ROTOR = False

SATURATION_ENABLED = True

_header("СРАВНЕНИЕ: С НАСЫЩЕНИЕМ  vs  БЕЗ НАСЫЩЕНИЯ")

def compare_val(name, v_sat, v_lin, unit=""):
    diff = v_sat - v_lin
    if abs(v_lin) > 1e-10:
        pct = diff / abs(v_lin) * 100
        print(f"    {name:30s}  {v_sat:10.2f}  {v_lin:10.2f}  {diff:+10.2f}  ({pct:+.1f}%) {unit}")
    else:
        print(f"    {name:30s}  {v_sat:10.2f}  {v_lin:10.2f}  {diff:+10.2f}  {unit}")

idx_s = np.searchsorted(res_sat['t'], 1.49)
idx_l = np.searchsorted(res_lin['t'], 1.49)

_section("ХХ (t ~= 1.49 с)")
_compare_header()
compare_val("n, об/мин", res_sat['n_rpm'][idx_s], res_lin['n_rpm'][idx_l])
compare_val("Me, Н*м", res_sat['Me'][idx_s], res_lin['Me'][idx_l])
compare_val("Is, А", res_sat['Is_rms'][idx_s], res_lin['Is_rms'][idx_l])
compare_val("Im_dq, А", res_sat['Im'][idx_s], res_lin['Im'][idx_l])
compare_val("Lm_3ph, мГн", res_sat['Lm'][idx_s]*1e3, res_lin['Lm'][idx_l]*1e3)
compare_val("i_n, А", res_sat['i_neutral'][idx_s], res_lin['i_neutral'][idx_l])

_section("Нагрузка (конец)")
_compare_header()
compare_val("n, об/мин", res_sat['n_rpm'][-1], res_lin['n_rpm'][-1])
compare_val("Me, Н*м", res_sat['Me'][-1], res_lin['Me'][-1])
compare_val("Is, А", res_sat['Is_rms'][-1], res_lin['Is_rms'][-1])
compare_val("Ps, кВт", res_sat['Ps'][-1]/1e3, res_lin['Ps'][-1]/1e3)
compare_val("Pmech, кВт", res_sat['Pmech'][-1]/1e3, res_lin['Pmech'][-1]/1e3)
compare_val("s", res_sat['slip'][-1], res_lin['slip'][-1])
compare_val("i_n, А", res_sat['i_neutral'][-1], res_lin['i_neutral'][-1])

_section("Пуск")
_compare_header()
compare_val("Max|Me|, Н*м", np.max(np.abs(res_sat['Me'])), np.max(np.abs(res_lin['Me'])))
compare_val("Max|iA|, А", np.max(np.abs(res_sat['iA'])), np.max(np.abs(res_lin['iA'])))
compare_val("Max|i_n|, А", np.max(np.abs(res_sat['i_neutral'])), np.max(np.abs(res_lin['i_neutral'])))

t_s = res_sat['t']
t_l = res_lin['t']
t_ms_s = t_s * 1000
t_ms_l = t_l * 1000

plt.rcParams.update({
    'font.size': 9, 'figure.dpi': 150, 'lines.linewidth': 0.5,
    'axes.grid': True, 'grid.alpha': 0.3,
})
cs = ['#d62728', '#1f77b4', '#2ca02c']
t_ld = 1500

def vline(ax): ax.axvline(t_ld, c='red', lw=0.5, ls=':', alpha=0.5)

fig_cmp = plt.figure(figsize=(17, 28))
fig_cmp.suptitle(
    f'СРАВНЕНИЕ: С насыщением vs Без насыщения ({CONN_MODE})',
    fontsize=13, fontweight='bold', y=0.995)
gs_c = GridSpec(7, 2, hspace=0.45, wspace=0.25,
                left=0.06, right=0.97, top=0.96, bottom=0.03)

ax = fig_cmp.add_subplot(gs_c[0, 0])
ax.plot(t_ms_s, res_sat['iA'], c='#d62728', lw=0.4, label='$i_A$ насыщ')
ax.plot(t_ms_l, res_lin['iA'], c='#1f77b4', lw=0.3, ls='--', label='$i_A$ линейн')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Ток фазы A'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[0, 1])
ax.plot(t_ms_s, res_sat['ia'], c='#d62728', lw=0.4, label='$i_a$ насыщ')
ax.plot(t_ms_l, res_lin['ia'], c='#1f77b4', lw=0.3, ls='--', label='$i_a$ линейн')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Ток ротора фаза a'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[1, 0])
ax.plot(t_ms_s, res_sat['Me'], c='#d62728', lw=0.35, label='насыщ')
ax.plot(t_ms_l, res_lin['Me'], c='#1f77b4', lw=0.3, ls='--', label='линейн')
ax.axhline(1096, c='gray', ls=':', lw=0.5)
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$M_e$, Н*м')
ax.set_title('Электромагнитный момент'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[1, 1])
ax.plot(t_ms_s, res_sat['n_rpm'], c='#d62728', lw=0.5, label='насыщ')
ax.plot(t_ms_l, res_lin['n_rpm'], c='#1f77b4', lw=0.4, ls='--', label='линейн')
ax.axhline(n_sync, c='gray', ls=':', lw=0.8)
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('n, об/мин')
ax.set_title('Скорость'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[2, 0])
ax.plot(t_ms_s, res_sat['PsiA'], c='#d62728', lw=0.5, label='$\\Psi_A$ насыщ')
ax.plot(t_ms_l, res_lin['PsiA'], c='#1f77b4', lw=0.4, ls='--', label='$\\Psi_A$ линейн')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('Psi, Вб')
ax.set_title('Потокосцепление статора A'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[2, 1])
ax.plot(t_ms_s, res_sat['Lm']*1e3, c='#d62728', lw=0.5, label='$L_m$ насыщ')
ax.plot(t_ms_l, res_lin['Lm']*1e3, c='#1f77b4', lw=0.4, ls='--', label='$L_m$ линейн (const)')
ax.axhline(Lm_nom*1e3, c='gray', ls=':', lw=0.8)
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$L_m$, мГн')
ax.set_title('Взаимная индуктивность'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[3, 0])
ax.plot(t_ms_s, res_sat['Is_rms'], c='#d62728', lw=0.4, label='насыщ')
ax.plot(t_ms_l, res_lin['Is_rms'], c='#1f77b4', lw=0.3, ls='--', label='линейн')
ax.axhline(57.45, c='green', ls=':', lw=0.8, label='$I_0$=57.45')
ax.axhline(204.42, c='gray', ls=':', lw=0.8, label='$I_n$=204.42')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_s$, А')
ax.set_title('Действующий ток статора'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[3, 1])
ax.plot(t_ms_s, res_sat['Im'], c='#d62728', lw=0.5, label='насыщ')
ax.plot(t_ms_l, res_lin['Im'], c='#1f77b4', lw=0.4, ls='--', label='линейн')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_m$ (dq), А')
ax.set_title('Ток намагничивания'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[4, 0])
ax.plot(t_ms_s, res_sat['Ps']/1e3, c='#d62728', lw=0.3, label='$P_{эл}$ насыщ')
ax.plot(t_ms_l, res_lin['Ps']/1e3, c='#1f77b4', lw=0.3, ls='--', label='$P_{эл}$ линейн')
ax.axhline(170, c='gray', ls=':', lw=0.8)
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('P, кВт')
ax.set_title('Электрическая мощность'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[4, 1])
ax.plot(t_ms_s, res_sat['Pmech']/1e3, c='#d62728', lw=0.3, label='$P_{мех}$ насыщ')
ax.plot(t_ms_l, res_lin['Pmech']/1e3, c='#1f77b4', lw=0.3, ls='--', label='$P_{мех}$ линейн')
ax.axhline(170, c='gray', ls=':', lw=0.8)
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('P, кВт')
ax.set_title('Механическая мощность'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[5, 0])
ax.plot(t_ms_s, res_sat['i_neutral'], c='#d62728', lw=0.5, label='$i_n$ насыщ')
ax.plot(t_ms_l, res_lin['i_neutral'], c='#1f77b4', lw=0.4, ls='--', label='$i_n$ линейн')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title('Ток нейтрали'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[5, 1])
ax.plot(t_ms_s, res_sat['slip'], c='#d62728', lw=0.5, label='насыщ')
ax.plot(t_ms_l, res_lin['slip'], c='#1f77b4', lw=0.4, ls='--', label='линейн')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('s')
ax.set_title('Скольжение'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[6, 0])
ax.plot(res_sat['n_rpm'], res_sat['Me'], c='#d62728', lw=0.15, alpha=0.6, label='насыщ')
ax.plot(res_lin['n_rpm'], res_lin['Me'], c='#1f77b4', lw=0.15, alpha=0.6, label='линейн')
ax.axhline(0, c='k', lw=0.3)
ax.set_xlabel('n, об/мин'); ax.set_ylabel('$M_e$, Н*м')
ax.set_title('Механическая характеристика'); ax.legend(fontsize=7)

ax = fig_cmp.add_subplot(gs_c[6, 1])
m_start_s = res_sat['t'] <= 0.2
m_start_l = res_lin['t'] <= 0.2
ax.plot(res_sat['t'][m_start_s]*1e3, res_sat['iA'][m_start_s], c='#d62728', lw=0.5, label='$i_A$ насыщ')
ax.plot(res_lin['t'][m_start_l]*1e3, res_lin['iA'][m_start_l], c='#1f77b4', lw=0.4, ls='--', label='$i_A$ линейн')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Пусковой ток $i_A$ (0-200 мс)'); ax.legend(fontsize=7)

plt.savefig('adm174_comparison_sat_vs_lin.png', dpi=150, bbox_inches='tight')
_fig_saved("СРАВНЕНИЕ")

t_arr = res_sat['t']
t_ms = t_arr * 1000
iA = res_sat['iA']; iB = res_sat['iB']; iC = res_sat['iC']
ia = res_sat['ia']; ib = res_sat['ib']; ic = res_sat['ic']
Me = res_sat['Me']; Lm_arr = res_sat['Lm']; Im_arr = res_sat['Im']
n_rpm = res_sat['n_rpm']; slip = res_sat['slip']
PsiA = res_sat['PsiA']; PsiB = res_sat['PsiB']; PsiC = res_sat['PsiC']
Psi_a = res_sat['Psi_a']; Psi_b = res_sat['Psi_b']; Psi_c = res_sat['Psi_c']
Is_rms = res_sat['Is_rms']; Ps = res_sat['Ps']; Pmech = res_sat['Pmech']
is0 = res_sat['is0']; i_neutral = res_sat['i_neutral']
us0 = res_sat['us0']; u_neutral = res_sat['u_neutral']

fig = plt.figure(figsize=(17, 26))
fig.suptitle(
    f'АД ADM174 с насыщением (v5, {CONN_MODE})\n'
    f'Psi_m = {_a_mag:.2f}*atan({_b_mag}*Im),  '
    'нормировка Im -- power-invariant dq',
    fontsize=12, fontweight='bold', y=0.995)
gs = GridSpec(7, 2, hspace=0.45, wspace=0.25,
              left=0.06, right=0.97, top=0.96, bottom=0.03)

ax=fig.add_subplot(gs[0,0])
ax.plot(t_ms,iA,c=cs[0],label='$i_A$'); ax.plot(t_ms,iB,c=cs[1],label='$i_B$')
ax.plot(t_ms,iC,c=cs[2],label='$i_C$'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А'); ax.set_title('Токи статора'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[0,1])
ax.plot(t_ms,ia,c=cs[0],ls='--',label='$i_a$')
ax.plot(t_ms,ib,c=cs[1],ls='--',label='$i_b$')
ax.plot(t_ms,ic,c=cs[2],ls='--',label='$i_c$'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А'); ax.set_title('Токи ротора'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[1,0])
ax.plot(t_ms,Me,c='#9467bd',lw=0.35); ax.axhline(0,c='k',lw=0.3)
ax.axhline(1096,c='red',ls='--',lw=0.5,alpha=0.5,label='$M_c$=1096'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$M_e$, Н*м'); ax.set_title('Момент'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[1,1])
ax.plot(t_ms,n_rpm,c='#ff7f0e',lw=0.7)
ax.axhline(n_sync,c='gray',ls=':',lw=0.8,label=f'$n_0$={n_sync:.0f}'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('n, об/мин'); ax.set_title('Скорость'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[2,0])
ax.plot(t_ms,PsiA,c=cs[0],label='$\\Psi_A$'); ax.plot(t_ms,PsiB,c=cs[1],label='$\\Psi_B$')
ax.plot(t_ms,PsiC,c=cs[2],label='$\\Psi_C$'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('Psi, Вб'); ax.set_title('Psi статора'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[2,1])
ax.plot(t_ms,Psi_a,c=cs[0],ls='--',label='$\\Psi_a$')
ax.plot(t_ms,Psi_b,c=cs[1],ls='--',label='$\\Psi_b$')
ax.plot(t_ms,Psi_c,c=cs[2],ls='--',label='$\\Psi_c$'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('Psi, Вб'); ax.set_title('Psi ротора'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[3,0])
ax.plot(t_ms,slip,c='#8c564b',lw=0.5); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('s'); ax.set_title('Скольжение')

ax=fig.add_subplot(gs[3,1])
ax.plot(n_rpm,Me,c='#9467bd',lw=0.15,alpha=0.6); ax.axhline(0,c='k',lw=0.3)
ax.set_xlabel('n, об/мин'); ax.set_ylabel('$M_e$, Н*м'); ax.set_title('Мех. характеристика')

ax=fig.add_subplot(gs[4,0])
ax.plot(t_ms,Lm_arr*1e3,c='#e377c2',lw=0.5)
ax.axhline(Lm_nom*1e3,c='gray',ls=':',lw=0.8,label=f'ном {Lm_nom*1e3:.2f} мГн'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$L_m$, мГн'); ax.set_title('$L_m(I_m)$'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[4,1])
ax.plot(t_ms,Im_arr,c='#17becf',lw=0.5); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_m$ (dq), А'); ax.set_title('Ток намагничивания')

ax=fig.add_subplot(gs[5,0])
ax.plot(t_ms,Is_rms,c='#17becf',lw=0.4)
ax.axhline(57.45,c='green',ls=':',lw=0.8,label='$I_0$=57.45')
ax.axhline(204.42,c='gray',ls=':',lw=0.8,label='$I_n$=204.42'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_s$, А'); ax.set_title('Действ. ток'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[5,1])
ax.plot(t_ms,Ps/1e3,c='#e377c2',lw=0.3,label='$P_{эл}$')
ax.plot(t_ms,Pmech/1e3,c='#bcbd22',lw=0.3,label='$P_{мех}$')
ax.axhline(170,c='gray',ls=':',lw=0.8,label='$P_{2n}$=170'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('P, кВт'); ax.set_title('Мощности'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[6,0])
ax.plot(t_ms, is0, c='#ff6600', lw=0.5, label='$i_s^{(0)}$')
ax.plot(t_ms, i_neutral, c='#9467bd', lw=0.3, ls='--', label='$i_n = \\sqrt{3} \\cdot i_s^{(0)}$')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('А')
ax.set_title(f'Нулевая составляющая тока ({CONN_MODE})')
ax.legend(fontsize=7)

ax=fig.add_subplot(gs[6,1])
if CONN_MODE == 'star_isolated':
    ax.plot(t_ms, u_neutral, c='#2ca02c', lw=0.5, label='$u_n$')
    ax.plot(t_ms, us0, c='#d62728', lw=0.3, ls='--', label='$u_s^{(0)} = \\sqrt{3} \\cdot u_n$')
    ax.set_ylabel('В')
    ax.set_title('Напряжение нейтрали и $u_s^{(0)}$')
    ax.legend(fontsize=7)
elif CONN_MODE == 'star_grounded':
    ax.plot(t_ms, i_neutral, c='#9467bd', lw=0.5, label='$i_n = i_A+i_B+i_C$')
    ax.set_ylabel('А')
    ax.set_title('Ток нейтрали (заземлённая нейтраль)')
    ax.legend(fontsize=7)
vline(ax)
ax.set_xlabel('t, мс')

plt.savefig('adm174_sat_full.png', dpi=150, bbox_inches='tight')
_fig_saved("1 (полная картина)")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(f'Кривая намагничивания', fontsize=12, fontweight='bold')
Im_p = np.linspace(0.1, 3000, 500)
Psi_p = psi_m_dq(Im_p)
Lm_dq_p = Psi_p / Im_p * 1e3
ax1.plot(Im_p,Psi_p,'b-',lw=1.5)
ax1.axhline(_Psi_max_dq,c='gray',ls=':',lw=0.8,label=f'Предел: {_Psi_max_dq:.2f} Вб')
ax1.set_xlabel('$I_m$ (dq), А'); ax1.set_ylabel('$\\Psi_m$ (dq), Вб')
ax1.set_title('$\\Psi_m(I_m)$'); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(Im_p,Lm_dq_p,'b-',lw=1.5)
ax2.axhline(Lm_dq_nom*1e3,c='gray',ls=':',lw=0.8,label=f'Ном. {Lm_dq_nom*1e3:.2f} мГн')
for Im_mark, label in [(57, 'ХХ'), (200, 'Ном'), (800, 'Пуск')]:
    Lm_val = Lm_dq_of_Im(Im_mark) * 1e3
    ax2.plot(Im_mark, Lm_val, 'ro', ms=6)
    ax2.annotate(f'{label}\n{Lm_val:.1f}', (Im_mark, Lm_val),
                 textcoords="offset points", xytext=(10, 5), fontsize=8)
ax2.set_xlabel('$I_m$ (dq), А'); ax2.set_ylabel('$L_m$ (dq), мГн')
ax2.set_title('$L_m(I_m)$'); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('adm174_sat_mag.png', dpi=150, bbox_inches='tight')
_fig_saved("2 (кривая намагничивания)")

fig3, axes = plt.subplots(2, 2, figsize=(15, 9))
fig3.suptitle(f'Пуск с насыщением (v5, {CONN_MODE}, 0-200 мс)', fontsize=12, fontweight='bold')
m = t_arr <= 0.2
axes[0,0].plot(t_arr[m]*1e3,iA[m],c=cs[0],lw=0.5,label='$i_A$')
axes[0,0].plot(t_arr[m]*1e3,iB[m],c=cs[1],lw=0.5,label='$i_B$')
axes[0,0].plot(t_arr[m]*1e3,iC[m],c=cs[2],lw=0.5,label='$i_C$')
axes[0,0].set_title('Пуск. токи'); axes[0,0].legend(fontsize=7); axes[0,0].grid(alpha=0.3)
axes[0,1].plot(t_arr[m]*1e3,Lm_arr[m]*1e3,c='#e377c2',lw=0.7)
axes[0,1].axhline(Lm_nom*1e3,c='gray',ls=':'); axes[0,1].set_title('Lm при пуске'); axes[0,1].grid(alpha=0.3)
axes[1,0].plot(t_arr[m]*1e3,Me[m],c='#9467bd',lw=0.4); axes[1,0].set_title('Пуск. момент'); axes[1,0].grid(alpha=0.3)
axes[1,1].plot(t_arr[m]*1e3,i_neutral[m],c='#ff6600',lw=0.5)
axes[1,1].set_title(f'Ток нейтрали при пуске ({CONN_MODE})'); axes[1,1].set_ylabel('$i_n$, А'); axes[1,1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('adm174_sat_startup.png', dpi=150, bbox_inches='tight')
_fig_saved("3 (пуск)")

fig4, axes4 = plt.subplots(2, 2, figsize=(15, 9))
fig4.suptitle(f'Наброс 1096 Н*м (v5, {CONN_MODE})', fontsize=12, fontweight='bold')
m4 = (t_arr >= 1.4) & (t_arr <= 2.5)
axes4[0,0].plot(t_arr[m4]*1e3,iA[m4],c=cs[0],lw=0.4,label='$i_A$')
axes4[0,0].plot(t_arr[m4]*1e3,iB[m4],c=cs[1],lw=0.4,label='$i_B$')
axes4[0,0].plot(t_arr[m4]*1e3,iC[m4],c=cs[2],lw=0.4,label='$i_C$')
axes4[0,0].axvline(1500,c='red',lw=0.5,ls=':'); axes4[0,0].legend(fontsize=7); axes4[0,0].grid(alpha=0.3)
axes4[0,1].plot(t_arr[m4]*1e3,Me[m4],c='#9467bd',lw=0.4)
axes4[0,1].axhline(1096,c='red',ls='--',lw=0.5); axes4[0,1].axvline(1500,c='red',lw=0.5,ls=':'); axes4[0,1].grid(alpha=0.3)
axes4[1,0].plot(t_arr[m4]*1e3,i_neutral[m4],c='#ff6600',lw=0.5)
axes4[1,0].axvline(1500,c='red',lw=0.5,ls=':'); axes4[1,0].set_ylabel('$i_n$, А')
axes4[1,0].set_title(f'Ток нейтрали ({CONN_MODE})'); axes4[1,0].grid(alpha=0.3)
axes4[1,1].plot(t_arr[m4]*1e3,n_rpm[m4],c='#ff7f0e',lw=0.7)
axes4[1,1].axhline(n_sync,c='gray',ls=':'); axes4[1,1].axvline(1500,c='red',lw=0.5,ls=':'); axes4[1,1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('adm174_sat_load.png', dpi=150, bbox_inches='tight')
_fig_saved("4 (наброс нагрузки)")

T_period = 1.0 / f


def harmonic_analysis(t_sig, x_sig, f_fund, n_periods, n_harmonics=30):
    """
    БПФ сигнала на окне ровно n_periods периодов (от конца сигнала назад).
    Интерполируем на равномерную сетку (данные Radau неравномерны).
    """
    T = n_periods / f_fund
    t_start = t_sig[-1] - T
    mask = t_sig >= t_start
    t_w = t_sig[mask]
    x_w = x_sig[mask]
    N_pts = 4096
    t_uni = np.linspace(t_w[0], t_w[-1], N_pts, endpoint=False)
    x_uni = np.interp(t_uni, t_w, x_w)
    dt = t_uni[1] - t_uni[0]
    X = np.fft.rfft(x_uni)
    freqs = np.fft.rfftfreq(N_pts, d=dt)
    amps = 2.0 * np.abs(X) / N_pts
    amps[0] /= 2.0
    harmonics = []
    for k in range(0, n_harmonics + 1):
        f_target = k * f_fund
        idx = np.argmin(np.abs(freqs - f_target))
        harmonics.append([k, freqs[idx], amps[idx]])
    harmonics = np.array(harmonics)
    A1 = harmonics[1, 2]
    if A1 > 1e-10:
        thd = np.sqrt(np.sum(harmonics[2:, 2]**2)) / A1 * 100
    else:
        thd = 0.0
    return t_uni, x_uni, harmonics, thd


t_xx_end = 1.5
mask_xx = t_arr < t_xx_end
t_xx = t_arr[mask_xx]

t_w_xxA, iA_w_xx, harm_xxA, thd_xxA = harmonic_analysis(t_xx, iA[mask_xx], f, 4)
t_w_xxB, iB_w_xx, harm_xxB, thd_xxB = harmonic_analysis(t_xx, iB[mask_xx], f, 4)
t_w_xxC, iC_w_xx, harm_xxC, thd_xxC = harmonic_analysis(t_xx, iC[mask_xx], f, 4)
t_w_xx0, in_w_xx, harm_xx0, thd_xx0 = harmonic_analysis(t_xx, i_neutral[mask_xx], f, 4)

t_w_ldA, iA_w_ld, harm_ldA, thd_ldA = harmonic_analysis(t_arr, iA, f, 4)
t_w_ldB, iB_w_ld, harm_ldB, thd_ldB = harmonic_analysis(t_arr, iB, f, 4)
t_w_ldC, iC_w_ld, harm_ldC, thd_ldC = harmonic_analysis(t_arr, iC, f, 4)
t_w_ld0, in_w_ld, harm_ld0, thd_ld0 = harmonic_analysis(t_arr, i_neutral, f, 4)

phase_names = ['$i_A$', '$i_B$', '$i_C$']
phase_colors = ['#d62728', '#1f77b4', '#2ca02c']

harm_xx_phases = [harm_xxA, harm_xxB, harm_xxC]
thd_xx_phases = [thd_xxA, thd_xxB, thd_xxC]
t_w_xx_phases = [t_w_xxA, t_w_xxB, t_w_xxC]
i_w_xx_phases = [iA_w_xx, iB_w_xx, iC_w_xx]

harm_ld_phases = [harm_ldA, harm_ldB, harm_ldC]
thd_ld_phases = [thd_ldA, thd_ldB, thd_ldC]
t_w_ld_phases = [t_w_ldA, t_w_ldB, t_w_ldC]
i_w_ld_phases = [iA_w_ld, iB_w_ld, iC_w_ld]

fig5a = plt.figure(figsize=(17, 20))
fig5a.suptitle(f'Гармонический анализ токов статора -- ХХ ({CONN_MODE})',
               fontsize=13, fontweight='bold', y=0.98)
gs5a = GridSpec(4, 2, hspace=0.40, wspace=0.30,
                left=0.07, right=0.97, top=0.93, bottom=0.04)

for ph_idx in range(3):
    ax = fig5a.add_subplot(gs5a[ph_idx, 0])
    t_w = t_w_xx_phases[ph_idx]
    ax.plot((t_w - t_w[0])*1e3, i_w_xx_phases[ph_idx], c=phase_colors[ph_idx], lw=0.8)
    ax.set_xlabel('t, мс'); ax.set_ylabel(f'{phase_names[ph_idx]}, А')
    ax.set_title(f'ХХ -- форма тока {phase_names[ph_idx]} (4 периода до t=1.5 с)')
    ax.grid(alpha=0.3)

    ax = fig5a.add_subplot(gs5a[ph_idx, 1])
    harm = harm_xx_phases[ph_idx]
    k_plot = harm[1:, 0].astype(int)
    a_plot = harm[1:, 2]
    ax.bar(k_plot, a_plot, color=phase_colors[ph_idx], width=0.7, alpha=0.8)
    A1 = harm[1, 2]
    for i, (k, a) in enumerate(zip(k_plot, a_plot)):
        if a > 0.01 * A1:
            ax.text(k, a + A1*0.02, f'{a:.1f}',
                    ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xlabel('Номер гармоники'); ax.set_ylabel('Амплитуда, А')
    ax.set_title(f'ХХ -- спектр {phase_names[ph_idx]} (THD = {thd_xx_phases[ph_idx]:.2f}%)')
    ax.set_xlim(0, 25); ax.grid(alpha=0.3, axis='y')

ax = fig5a.add_subplot(gs5a[3, 0])
ax.plot((t_w_xx0 - t_w_xx0[0])*1e3, in_w_xx, c='#ff6600', lw=0.8)
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title(f'ХХ -- ток нейтрали $i_n$ (4 периода)')
ax.grid(alpha=0.3)

ax = fig5a.add_subplot(gs5a[3, 1])
k_plot0 = harm_xx0[1:, 0].astype(int)
a_plot0 = harm_xx0[1:, 2]
ax.bar(k_plot0, a_plot0, color='#ff6600', width=0.7, alpha=0.8)
max_a0 = np.max(a_plot0) if len(a_plot0) > 0 else 1.0
for i, (k, a) in enumerate(zip(k_plot0, a_plot0)):
    if a > 0.01 * max_a0:
        ax.text(k, a + max_a0*0.02, f'{a:.2f}',
                ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_xlabel('Номер гармоники'); ax.set_ylabel('Амплитуда, А')
ax.set_title(f'ХХ -- спектр $i_n$ ')
ax.set_xlim(0, 25); ax.grid(alpha=0.3, axis='y')

plt.savefig('adm174_sat_harmonics_xx.png', dpi=150, bbox_inches='tight')
_fig_saved("5a (гармоники ХХ)")

fig5b = plt.figure(figsize=(17, 20))
fig5b.suptitle(f'Гармонический анализ токов статора -- Нагрузка ({CONN_MODE})',
               fontsize=13, fontweight='bold', y=0.98)
gs5b = GridSpec(4, 2, hspace=0.40, wspace=0.30,
                left=0.07, right=0.97, top=0.93, bottom=0.04)

for ph_idx in range(3):
    ax = fig5b.add_subplot(gs5b[ph_idx, 0])
    t_w = t_w_ld_phases[ph_idx]
    ax.plot((t_w - t_w[0])*1e3, i_w_ld_phases[ph_idx], c=phase_colors[ph_idx], lw=0.8)
    ax.set_xlabel('t, мс'); ax.set_ylabel(f'{phase_names[ph_idx]}, А')
    ax.set_title(f'Нагрузка -- форма тока {phase_names[ph_idx]} (последние 4 периода)')
    ax.grid(alpha=0.3)

    ax = fig5b.add_subplot(gs5b[ph_idx, 1])
    harm = harm_ld_phases[ph_idx]
    k_plot = harm[1:, 0].astype(int)
    a_plot = harm[1:, 2]
    ax.bar(k_plot, a_plot, color=phase_colors[ph_idx], width=0.7, alpha=0.8)
    A1 = harm[1, 2]
    for i, (k, a) in enumerate(zip(k_plot, a_plot)):
        if a > 0.01 * A1:
            ax.text(k, a + A1*0.02, f'{a:.1f}',
                    ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xlabel('Номер гармоники'); ax.set_ylabel('Амплитуда, А')
    ax.set_title(f'Нагрузка -- спектр {phase_names[ph_idx]} (THD = {thd_ld_phases[ph_idx]:.2f}%)')
    ax.set_xlim(0, 25); ax.grid(alpha=0.3, axis='y')

ax = fig5b.add_subplot(gs5b[3, 0])
ax.plot((t_w_ld0 - t_w_ld0[0])*1e3, in_w_ld, c='#ff6600', lw=0.8)
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title(f'Нагрузка -- ток нейтрали $i_n$ (последние 4 периода)')
ax.grid(alpha=0.3)

ax = fig5b.add_subplot(gs5b[3, 1])
k_plot0_ld = harm_ld0[1:, 0].astype(int)
a_plot0_ld = harm_ld0[1:, 2]
ax.bar(k_plot0_ld, a_plot0_ld, color='#d62728', width=0.7, alpha=0.8)
max_a0_ld = np.max(a_plot0_ld) if len(a_plot0_ld) > 0 else 1.0
for i, (k, a) in enumerate(zip(k_plot0_ld, a_plot0_ld)):
    if a > 0.01 * max_a0_ld:
        ax.text(k, a + max_a0_ld*0.02, f'{a:.2f}',
                ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_xlabel('Номер гармоники'); ax.set_ylabel('Амплитуда, А')
ax.set_title(f'Нагрузка -- спектр $i_n$')
ax.set_xlim(0, 25); ax.grid(alpha=0.3, axis='y')

plt.savefig('adm174_sat_harmonics_load.png', dpi=150, bbox_inches='tight')
_fig_saved("5b (гармоники нагрузка)")

_subheader(f"ГАРМОНИЧЕСКИЙ АНАЛИЗ токов iA, iB, iC и тока нейтрали i_n ({CONN_MODE})",
           ch='-', width=120)

_section("ХХ (4 периода до t=1.5 с)")
_harm_table_header_4col()
for k_idx in range(0, min(16, len(harm_xxA))):
    aA = harm_xxA[k_idx, 2]; aB = harm_xxB[k_idx, 2]
    aC = harm_xxC[k_idx, 2]; a0 = harm_xx0[k_idx, 2]
    _harm_table_row(k_idx, harm_xxA[k_idx,1], aA, aB, aC, a0)
_thd_line(thd_xxA, thd_xxB, thd_xxC, prefix="ХХ: ")
print(f"  ХХ: THD i_n = {thd_xx0:.2f}%")

_section("Нагрузка (последние 4 периода)")
_harm_table_header_4col()
for k_idx in range(0, min(16, len(harm_ldA))):
    aA = harm_ldA[k_idx, 2]; aB = harm_ldB[k_idx, 2]
    aC = harm_ldC[k_idx, 2]; a0 = harm_ld0[k_idx, 2]
    _harm_table_row(k_idx, harm_ldA[k_idx,1], aA, aB, aC, a0)
_thd_line(thd_ldA, thd_ldB, thd_ldC, prefix="Нагр: ")
print(f"  Нагр: THD i_n = {thd_ld0:.2f}%")

fig6, axes6 = plt.subplots(2, 2, figsize=(15, 9))
fig6.suptitle(f'Нулевые составляющие -- установившийся режим ({CONN_MODE})',
              fontsize=12, fontweight='bold')

mask_ss_xx = (t_arr >= 1.42) & (t_arr < 1.5)
ax = axes6[0, 0]
ax.plot(t_arr[mask_ss_xx]*1e3, iA[mask_ss_xx], c=cs[0], lw=0.8, label='$i_A$')
ax.plot(t_arr[mask_ss_xx]*1e3, is0[mask_ss_xx], c='#ff6600', lw=1.2, label='$i_s^{(0)}$')
ax.set_xlabel('t, мс'); ax.set_ylabel('А')
ax.set_title('ХХ: $i_A$ и $i_s^{(0)}$'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes6[0, 1]
ax.plot(t_arr[mask_ss_xx]*1e3, i_neutral[mask_ss_xx], c='#9467bd', lw=1.0)
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title('ХХ: ток нейтрали $i_n$'); ax.grid(alpha=0.3)

mask_ss_ld = t_arr >= (t_arr[-1] - 4*T_period)
ax = axes6[1, 0]
ax.plot(t_arr[mask_ss_ld]*1e3, iA[mask_ss_ld], c=cs[0], lw=0.8, label='$i_A$')
ax.plot(t_arr[mask_ss_ld]*1e3, is0[mask_ss_ld], c='#ff6600', lw=1.2, label='$i_s^{(0)}$')
ax.set_xlabel('t, мс'); ax.set_ylabel('А')
ax.set_title('Нагрузка: $i_A$ и $i_s^{(0)}$'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes6[1, 1]
ax.plot(t_arr[mask_ss_ld]*1e3, i_neutral[mask_ss_ld], c='#9467bd', lw=1.0)
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title('Нагрузка: ток нейтрали $i_n$'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('adm174_sat_zero_seq.png', dpi=150, bbox_inches='tight')
_fig_saved("6 (нулевые составляющие)")


def run_harmonic_comparison(res, label_prefix):
    """Гармонический анализ трёх фаз + нейтраль для одного прогона."""
    t_arr_ = res['t']
    t_xx_end_ = 1.5
    mask_xx_ = t_arr_ < t_xx_end_
    t_xx_ = t_arr_[mask_xx_]

    results = {}
    for mode_key, t_sig, signals in [
        ('xx', t_xx_, {
            'iA': res['iA'][mask_xx_], 'iB': res['iB'][mask_xx_],
            'iC': res['iC'][mask_xx_], 'i_n': res['i_neutral'][mask_xx_]}),
        ('ld', t_arr_, {
            'iA': res['iA'], 'iB': res['iB'],
            'iC': res['iC'], 'i_n': res['i_neutral']}),
    ]:
        for sig_name, sig_data in signals.items():
            _, _, harm, thd = harmonic_analysis(t_sig, sig_data, f, 4)
            results[f'{mode_key}_{sig_name}_harm'] = harm
            results[f'{mode_key}_{sig_name}_thd'] = thd
    return results


harm_sat = run_harmonic_comparison(res_sat, "Насыщ")
harm_lin = run_harmonic_comparison(res_lin, "Линейн")

_header("СРАВНЕНИЕ ГАРМОНИК: С насыщением vs Без насыщения", ch='-', width=120)

for mode_key, mode_label in [('xx', 'ХХ'), ('ld', 'Нагрузка')]:
    _section(mode_label)
    print(f"  {'Гарм':>5}  {'f,Гц':>6}  "
          f"{'iA нас':>8}  {'iA лин':>8}  "
          f"{'iB нас':>8}  {'iB лин':>8}  "
          f"{'iC нас':>8}  {'iC лин':>8}  "
          f"{'in нас':>8}  {'in лин':>8}")
    h_sA = harm_sat[f'{mode_key}_iA_harm']
    h_lA = harm_lin[f'{mode_key}_iA_harm']
    h_sB = harm_sat[f'{mode_key}_iB_harm']
    h_lB = harm_lin[f'{mode_key}_iB_harm']
    h_sC = harm_sat[f'{mode_key}_iC_harm']
    h_lC = harm_lin[f'{mode_key}_iC_harm']
    h_sN = harm_sat[f'{mode_key}_i_n_harm']
    h_lN = harm_lin[f'{mode_key}_i_n_harm']
    for k_idx in range(min(16, len(h_sA))):
        print(f"  {k_idx:5d}  {h_sA[k_idx,1]:6.0f}  "
              f"{h_sA[k_idx,2]:8.2f}  {h_lA[k_idx,2]:8.2f}  "
              f"{h_sB[k_idx,2]:8.2f}  {h_lB[k_idx,2]:8.2f}  "
              f"{h_sC[k_idx,2]:8.2f}  {h_lC[k_idx,2]:8.2f}  "
              f"{h_sN[k_idx,2]:8.3f}  {h_lN[k_idx,2]:8.3f}")
    for ph in ['iA', 'iB', 'iC', 'i_n']:
        thd_s = harm_sat[f'{mode_key}_{ph}_thd']
        thd_l = harm_lin[f'{mode_key}_{ph}_thd']
        print(f"    THD {ph}: насыщ={thd_s:.2f}%,  линейн={thd_l:.2f}%,  разн={thd_s-thd_l:+.2f}%")


fig_hcmp = plt.figure(figsize=(17, 14))
fig_hcmp.suptitle('Сравнение спектров: насыщение vs линейная модель',
                   fontsize=13, fontweight='bold', y=0.98)
gs_h = GridSpec(2, 3, hspace=0.35, wspace=0.30,
                left=0.06, right=0.97, top=0.92, bottom=0.06)

phase_names_cmp = ['$i_A$', '$i_B$', '$i_C$']
phase_keys = ['iA', 'iB', 'iC']

for col, ph_key in enumerate(phase_keys):
    ax = fig_hcmp.add_subplot(gs_h[0, col])
    h_s = harm_sat[f'xx_{ph_key}_harm']
    h_l = harm_lin[f'xx_{ph_key}_harm']
    k_arr = h_s[1:16, 0].astype(int)
    width = 0.35
    ax.bar(k_arr - width/2, h_s[1:16, 2], width, color='#d62728', alpha=0.8, label='насыщ')
    ax.bar(k_arr + width/2, h_l[1:16, 2], width, color='#1f77b4', alpha=0.8, label='линейн')
    thd_s = harm_sat[f'xx_{ph_key}_thd']
    thd_l = harm_lin[f'xx_{ph_key}_thd']
    ax.set_xlabel('Номер гармоники'); ax.set_ylabel('А')
    ax.set_title(f'ХХ {phase_names_cmp[col]} (THD: {thd_s:.1f}% / {thd_l:.1f}%)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    ax = fig_hcmp.add_subplot(gs_h[1, col])
    h_s = harm_sat[f'ld_{ph_key}_harm']
    h_l = harm_lin[f'ld_{ph_key}_harm']
    ax.bar(k_arr - width/2, h_s[1:16, 2], width, color='#d62728', alpha=0.8, label='насыщ')
    ax.bar(k_arr + width/2, h_l[1:16, 2], width, color='#1f77b4', alpha=0.8, label='линейн')
    thd_s = harm_sat[f'ld_{ph_key}_thd']
    thd_l = harm_lin[f'ld_{ph_key}_thd']
    ax.set_xlabel('Номер гармоники'); ax.set_ylabel('А')
    ax.set_title(f'Нагр {phase_names_cmp[col]} (THD: {thd_s:.1f}% / {thd_l:.1f}%)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

plt.savefig('adm174_comparison_harmonics.png', dpi=150, bbox_inches='tight')
_fig_saved("СРАВНЕНИЕ гармоник")


_header("ЗАТОРМОЖЕННЫЙ РОТОР: СРАВНЕНИЕ С НАСЫЩЕНИЕМ vs БЕЗ НАСЫЩЕНИЯ")

_section("Установившийся режим (конец расчёта)")
_compare_header()
compare_val("Me уст., Н*м", res_lr_sat['Me'][-1], res_lr_lin['Me'][-1])
compare_val("Is уст., А", res_lr_sat['Is_rms'][-1], res_lr_lin['Is_rms'][-1])
compare_val("Ps уст., кВт", res_lr_sat['Ps'][-1]/1e3, res_lr_lin['Ps'][-1]/1e3)
compare_val("Lm_3ph, мГн", res_lr_sat['Lm'][-1]*1e3, res_lr_lin['Lm'][-1]*1e3)
compare_val("Im_dq, А", res_lr_sat['Im'][-1], res_lr_lin['Im'][-1])
compare_val("i_n уст., А", res_lr_sat['i_neutral'][-1], res_lr_lin['i_neutral'][-1])

_section("Переходный процесс")
_compare_header()
compare_val("Max|Me|, Н*м", np.max(np.abs(res_lr_sat['Me'])), np.max(np.abs(res_lr_lin['Me'])))
compare_val("Max|iA|, А", np.max(np.abs(res_lr_sat['iA'])), np.max(np.abs(res_lr_lin['iA'])))
compare_val("Max|Is|, А", np.max(res_lr_sat['Is_rms']), np.max(res_lr_lin['Is_rms']))
compare_val("Max|i_n|, А", np.max(np.abs(res_lr_sat['i_neutral'])), np.max(np.abs(res_lr_lin['i_neutral'])))

t_lr_s = res_lr_sat['t']
t_lr_l = res_lr_lin['t']
t_ms_lr_s = t_lr_s * 1000
t_ms_lr_l = t_lr_l * 1000

fig_lr = plt.figure(figsize=(17, 22))
fig_lr.suptitle(
    f'ЗАТОРМОЖЕННЫЙ РОТОР: С насыщением vs Без насыщения ({CONN_MODE})',
    fontsize=13, fontweight='bold', y=0.995)
gs_lr = GridSpec(5, 2, hspace=0.45, wspace=0.25,
                 left=0.06, right=0.97, top=0.96, bottom=0.03)

ax = fig_lr.add_subplot(gs_lr[0, 0])
ax.plot(t_ms_lr_s, res_lr_sat['iA'], c=cs[0], lw=0.5, label='$i_A$')
ax.plot(t_ms_lr_s, res_lr_sat['iB'], c=cs[1], lw=0.5, label='$i_B$')
ax.plot(t_ms_lr_s, res_lr_sat['iC'], c=cs[2], lw=0.5, label='$i_C$')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Токи статора (с насыщением)'); ax.legend(fontsize=7)

ax = fig_lr.add_subplot(gs_lr[0, 1])
ax.plot(t_ms_lr_l, res_lr_lin['iA'], c=cs[0], lw=0.5, ls='--', label='$i_A$')
ax.plot(t_ms_lr_l, res_lr_lin['iB'], c=cs[1], lw=0.5, ls='--', label='$i_B$')
ax.plot(t_ms_lr_l, res_lr_lin['iC'], c=cs[2], lw=0.5, ls='--', label='$i_C$')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Токи статора (без насыщения)'); ax.legend(fontsize=7)


ax = fig_lr.add_subplot(gs_lr[1, 0])
ax.plot(t_ms_lr_s, res_lr_sat['ia'], c=cs[0], lw=0.5, label='$i_a$ насыщ')
ax.plot(t_ms_lr_l, res_lr_lin['ia'], c=cs[1], lw=0.4, ls='--', label='$i_a$ линейн')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Ток ротора фаза a'); ax.legend(fontsize=7)


ax = fig_lr.add_subplot(gs_lr[1, 1])
ax.plot(t_ms_lr_s, res_lr_sat['Me'], c='#d62728', lw=0.5, label='насыщ')
ax.plot(t_ms_lr_l, res_lr_lin['Me'], c='#1f77b4', lw=0.4, ls='--', label='линейн')
ax.axhline(0, c='k', lw=0.3)
ax.set_xlabel('t, мс'); ax.set_ylabel('$M_e$, Н*м')
ax.set_title('Электромагнитный момент'); ax.legend(fontsize=7)


ax = fig_lr.add_subplot(gs_lr[2, 0])
ax.plot(t_ms_lr_s, res_lr_sat['PsiA'], c='#d62728', lw=0.5, label='$\\Psi_A$ насыщ')
ax.plot(t_ms_lr_l, res_lr_lin['PsiA'], c='#1f77b4', lw=0.4, ls='--', label='$\\Psi_A$ линейн')
ax.set_xlabel('t, мс'); ax.set_ylabel('Psi, Вб')
ax.set_title('Потокосцепление статора A'); ax.legend(fontsize=7)


ax = fig_lr.add_subplot(gs_lr[2, 1])
ax.plot(t_ms_lr_s, res_lr_sat['Lm']*1e3, c='#d62728', lw=0.5, label='$L_m$ насыщ')
ax.plot(t_ms_lr_l, res_lr_lin['Lm']*1e3, c='#1f77b4', lw=0.4, ls='--', label='$L_m$ линейн')
ax.axhline(Lm_nom*1e3, c='gray', ls=':', lw=0.8)
ax.set_xlabel('t, мс'); ax.set_ylabel('$L_m$, мГн')
ax.set_title('Взаимная индуктивность'); ax.legend(fontsize=7)


ax = fig_lr.add_subplot(gs_lr[3, 0])
ax.plot(t_ms_lr_s, res_lr_sat['Is_rms'], c='#d62728', lw=0.5, label='насыщ')
ax.plot(t_ms_lr_l, res_lr_lin['Is_rms'], c='#1f77b4', lw=0.4, ls='--', label='линейн')
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_s$, А')
ax.set_title('Действующий ток статора'); ax.legend(fontsize=7)


ax = fig_lr.add_subplot(gs_lr[3, 1])
ax.plot(t_ms_lr_s, res_lr_sat['Ps']/1e3, c='#d62728', lw=0.5, label='$P_{эл}$ насыщ')
ax.plot(t_ms_lr_l, res_lr_lin['Ps']/1e3, c='#1f77b4', lw=0.4, ls='--', label='$P_{эл}$ линейн')
ax.set_xlabel('t, мс'); ax.set_ylabel('P, кВт')
ax.set_title('Электрическая мощность (Pmech=0)'); ax.legend(fontsize=7)

ax = fig_lr.add_subplot(gs_lr[4, 0])
ax.plot(t_ms_lr_s, res_lr_sat['i_neutral'], c='#d62728', lw=0.5, label='$i_n$ насыщ')
ax.plot(t_ms_lr_l, res_lr_lin['i_neutral'], c='#1f77b4', lw=0.4, ls='--', label='$i_n$ линейн')
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title('Ток нейтрали'); ax.legend(fontsize=7)

ax = fig_lr.add_subplot(gs_lr[4, 1])
ax.plot(t_ms_lr_s, res_lr_sat['Im'], c='#d62728', lw=0.5, label='насыщ')
ax.plot(t_ms_lr_l, res_lr_lin['Im'], c='#1f77b4', lw=0.4, ls='--', label='линейн')
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_m$ (dq), А')
ax.set_title('Ток намагничивания'); ax.legend(fontsize=7)

plt.savefig('adm174_locked_rotor.png', dpi=150, bbox_inches='tight')
_fig_saved("ЗАТОРМОЖЕННЫЙ РОТОР")

fig_lr2, axes_lr2 = plt.subplots(2, 2, figsize=(15, 9))
fig_lr2.suptitle(f'Заторможенный ротор -- установившийся режим ({CONN_MODE})',
                 fontsize=12, fontweight='bold')

T_lr = 1.0 / f
mask_lr_ss = t_lr_s >= (t_lr_s[-1] - 4*T_lr)

ax = axes_lr2[0, 0]
ax.plot(t_lr_s[mask_lr_ss]*1e3, res_lr_sat['iA'][mask_lr_ss], c=cs[0], lw=0.8, label='$i_A$')
ax.plot(t_lr_s[mask_lr_ss]*1e3, res_lr_sat['iB'][mask_lr_ss], c=cs[1], lw=0.8, label='$i_B$')
ax.plot(t_lr_s[mask_lr_ss]*1e3, res_lr_sat['iC'][mask_lr_ss], c=cs[2], lw=0.8, label='$i_C$')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Токи статора (уст., насыщ)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes_lr2[0, 1]
ax.plot(t_lr_s[mask_lr_ss]*1e3, res_lr_sat['ia'][mask_lr_ss], c=cs[0], lw=0.8, label='$i_a$')
ax.plot(t_lr_s[mask_lr_ss]*1e3, res_lr_sat['ib'][mask_lr_ss], c=cs[1], lw=0.8, label='$i_b$')
ax.plot(t_lr_s[mask_lr_ss]*1e3, res_lr_sat['ic'][mask_lr_ss], c=cs[2], lw=0.8, label='$i_c$')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Токи ротора (уст., насыщ)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes_lr2[1, 0]
ax.plot(t_lr_s[mask_lr_ss]*1e3, res_lr_sat['Me'][mask_lr_ss], c='#9467bd', lw=0.8)
ax.axhline(0, c='k', lw=0.3)
ax.set_xlabel('t, мс'); ax.set_ylabel('$M_e$, Н*м')
ax.set_title('Момент (уст., насыщ)'); ax.grid(alpha=0.3)

ax = axes_lr2[1, 1]
ax.plot(t_lr_s[mask_lr_ss]*1e3, res_lr_sat['i_neutral'][mask_lr_ss], c='#ff6600', lw=0.8)
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title(f'Ток нейтрали (уст., насыщ, {CONN_MODE})'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('adm174_locked_rotor_steady.png', dpi=150, bbox_inches='tight')
_fig_saved("ЗАТОРМОЖЕННЫЙ РОТОР (уст. режим)")

_subheader(f"ГАРМОНИЧЕСКИЙ АНАЛИЗ -- ЗАТОРМОЖЕННЫЙ РОТОР ({CONN_MODE})", ch='-', width=80)

for res_lr, lr_label in [(res_lr_sat, 'С насыщением'), (res_lr_lin, 'Без насыщения')]:
    _section(f"{lr_label} (последние 4 периода)")
    t_lr_arr = res_lr['t']
    _, _, harm_lrA, thd_lrA = harmonic_analysis(t_lr_arr, res_lr['iA'], f, 4)
    _, _, harm_lrB, thd_lrB = harmonic_analysis(t_lr_arr, res_lr['iB'], f, 4)
    _, _, harm_lrC, thd_lrC = harmonic_analysis(t_lr_arr, res_lr['iC'], f, 4)
    _, _, harm_lrN, thd_lrN = harmonic_analysis(t_lr_arr, res_lr['i_neutral'], f, 4)

    _harm_table_header_4col()
    for k_idx in range(0, min(16, len(harm_lrA))):
        aA = harm_lrA[k_idx, 2]; aB = harm_lrB[k_idx, 2]
        aC = harm_lrC[k_idx, 2]; aN = harm_lrN[k_idx, 2]
        _harm_table_row(k_idx, harm_lrA[k_idx,1], aA, aB, aC, aN)
    _thd_line(thd_lrA, thd_lrB, thd_lrC)
    print(f"  THD i_n = {thd_lrN:.2f}%")

_header("Готово.")