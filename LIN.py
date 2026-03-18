"""
Исправления v5:
  A) Учёт нулевых составляющих токов и напряжений
     по методике Дроздовского-Дуды.
  B) Два режима подключения статора:
       'star_grounded'  -- звезда с заземлённой нейтралью
       'star_isolated'  -- звезда без нейтрали (iA+iB+iC = 0)
  C) Расчёт и визуализация:
       i_s^(0) = (iA+iB+iC)/sqrt3  -- нулевая составляющая тока
       u_s^(0) = (uA+uB+uC)/sqrt3  -- нулевая составляющая напряжения
       u_n = u_s^(0)/sqrt3          -- напряжение нейтрали
  D) Гармонический анализ нулевых составляющих.

Lm = const (номинальное значение)
"""

import numpy as np
from numpy.linalg import solve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time as timer


# Вспомогательные функции вывода

def _header(title, ch='=', width=72):
    line = ch * width
    print(f"\n{line}\n  {title}\n{line}")

def _subheader(title, ch='-', width=60):
    line = ch * width
    print(f"\n  {line}\n  {title}\n  {line}")

def _section(title):
    print(f"\n  === {title} ===")

def _fig_saved(name):
    print(f"  + Рис. {name} сохранён")

def _harm_table_header_4col():
    print(f"  {'Гарм':>5}  {'f, Гц':>8}  {'iA':>8}  {'iB':>8}  {'iC':>8}  {'i_n':>9}")

def _harm_table_row(k, freq, aA, aB, aC, aN):
    print(f"  {k:5d}  {freq:8.1f}  {aA:8.2f}  {aB:8.2f}  {aC:8.2f}  {aN:9.3f}")

def _thd_line(thd_A, thd_B, thd_C, prefix=""):
    print(f"  {prefix}THD iA = {thd_A:.2f}%,  THD iB = {thd_B:.2f}%,  THD iC = {thd_C:.2f}%")


CONN_MODE = 'star_grounded'


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

# Линейная модель: Lm = const = Lm_nom
Lm = Lm_nom

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


def _build_L(theta, Lm_val, out=None):
    """Сборка L 6x6 in-place."""
    Ls = Lls + Lm_val
    Lr = Llr + Lm_val
    hLm = Lm_val * 0.5
    cos_sr = Lm_val * np.cos(theta + _SR_ANGLES)

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
    Ток нейтрали: i_n = iA + iB + iC = sqrt3 * i_s^(0)
    """
    return _INV_SQRT3 * (i6[0] + i6[1] + i6[2])


def zero_seq_voltage(uA, uB, uC):
    """
    Нулевая составляющая напряжений статора:
      u_s^(0) = (uA + uB + uC) / sqrt3
    Напряжение нейтрали: u_n = u_s^(0) / sqrt3
    """
    return _INV_SQRT3 * (uA + uB + uC)


def neutral_voltage_from_zero_seq(u_s0):
    """u_n = u_s^(0) / sqrt3"""
    return u_s0 / _SQRT3


_L_buf = np.empty((6, 6))


def flux_to_currents(psi6, theta):
    """Прямое решение Psi = L(theta) * i. Lm = const."""
    _build_L(theta, Lm, _L_buf)
    i6 = solve(_L_buf, psi6)
    return i6


_L_buf_7 = np.empty((7, 7))


def flux_to_currents_isolated(psi6, theta):
    """
    Решение Psi = L(theta) * i с ограничением iA + iB + iC = 0.

    Расширяем систему 6x6 до 7x7 с множителем Лагранжа:
      [L  | c ] [i ]       = [Psi]
      [c^T | 0 ] [lambda_] = [0  ]
    где c = [1, 1, 1, 0, 0, 0]^T.
    """
    constraint = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    _build_L(theta, Lm, _L_buf)
    _L_buf_7[:6, :6] = _L_buf
    _L_buf_7[:6, 6] = constraint
    _L_buf_7[6, :6] = constraint
    _L_buf_7[6, 6] = 0.0
    rhs = np.zeros(7)
    rhs[:6] = psi6
    rhs[6] = 0.0
    sol7 = solve(_L_buf_7, rhs)
    return sol7[:6]


def electromagnetic_torque(i6, theta):
    """
    Электромагнитный момент:
      Me = p * is^T * (dLsr/dtheta) * ir
    dLsr/dtheta = -Lm * sin(theta + SR_ANGLES)
    """
    sin_sr = -Lm * np.sin(theta + _SR_ANGLES)
    is3 = i6[0:3]
    ir3 = i6[3:6]
    return p_poles * (is3 @ sin_sr @ ir3)


LOCKED_ROTOR = False

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
    """
    uA, uB, uC = supply_voltages_phase(t)
    return np.array([uA, uB, uC, 0.0, 0.0, 0.0])


def compute_neutral_voltage(psi6, i6, u6_source, theta, R_ph=None):
    """
    Вычисление напряжения нейтрали u_n для схемы star_isolated.

    u_n = (1/3)*sum(dpsi_dt_unconstrained[:3])
    где dpsi_unconstrained = u_source - R*i6
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
        i6 = flux_to_currents_isolated(psi6, theta_elec)
    else:
        i6 = flux_to_currents(psi6, theta_elec)

    u6 = supply_voltages(t)
    R_ph = R_phases_of_t(t)

    if CONN_MODE == 'star_isolated':
        u_n = compute_neutral_voltage(psi6, i6, u6, theta_elec, R_ph)
        u6[0] -= u_n
        u6[1] -= u_n
        u6[2] -= u_n

    dpsi_dt = u6 - R_ph * i6

    cos_sr = np.cos(theta_elec + _SR_ANGLES)
    Im = _compute_Im_dq(i6, cos_sr)
    Me_val = electromagnetic_torque(i6, theta_elec)

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
              f"Im={Im:.1f} А  "
              f"i0={i0:.2f} А  "
              f"calls={_call_count[0]:>9d}  wall={el:.0f}с", flush=True)
        _last_wall[0] = wt

    return np.concatenate([dpsi_dt, [domega_dt, dtheta_dt]])


def pq_decomposition(iA, iB, iC, uA, uB, uC):
    """
    Разложение тока статора на активную и реактивную составляющие
    по теории мгновенной мощности (Акаги).

    Вход: массивы мгновенных фазных токов и напряжений (длина Nt).
    Выход: словарь с массивами:
      - i_alpha, i_beta, u_alpha, u_beta  (Кларк)
      - p, q                              (мгновенные мощности)
      - iAp, iBp, iCp                     (активная составляющая, фазные)
      - iAq, iBq, iCq                     (реактивная составляющая, фазные)
      - i_alpha_p, i_beta_p               (активная, ab)
      - i_alpha_q, i_beta_q               (реактивная, ab)
      - Is_p_rms, Is_q_rms               (действ. значения составляющих)
      - cos_phi                           (мгновенный cosphi)
    """
    i_alpha = (2.0/3.0) * (iA - 0.5*iB - 0.5*iC)
    i_beta  = (2.0/3.0) * (_SQRT3_2) * (iB - iC)

    u_alpha = (2.0/3.0) * (uA - 0.5*uB - 0.5*uC)
    u_beta  = (2.0/3.0) * (_SQRT3_2) * (uB - uC)

    p = u_alpha * i_alpha + u_beta * i_beta   # активная
    q = u_beta * i_alpha - u_alpha * i_beta   # реактивная

    us2 = u_alpha**2 + u_beta**2
    # Защита от деления на ноль (в первые мгновения us2 ~ 0)
    us2_safe = np.where(us2 > 1e-6, us2, 1e-6)

    i_alpha_p = (u_alpha / us2_safe) * p
    i_beta_p  = (u_beta  / us2_safe) * p

    i_alpha_q = ( u_beta  / us2_safe) * q
    i_beta_q  = (-u_alpha / us2_safe) * q

    iAp = i_alpha_p
    iBp = -0.5 * i_alpha_p + _SQRT3_2 * i_beta_p
    iCp = -0.5 * i_alpha_p - _SQRT3_2 * i_beta_p

    iAq = i_alpha_q
    iBq = -0.5 * i_alpha_q + _SQRT3_2 * i_beta_q
    iCq = -0.5 * i_alpha_q - _SQRT3_2 * i_beta_q

    Is_p_rms = np.sqrt((iAp**2 + iBp**2 + iCp**2) / 3.0)
    Is_q_rms = np.sqrt((iAq**2 + iBq**2 + iCq**2) / 3.0)

    Is_total = np.sqrt(Is_p_rms**2 + Is_q_rms**2)
    cos_phi = np.where(Is_total > 1e-6, Is_p_rms / Is_total, 0.0)

    return dict(
        i_alpha=i_alpha, i_beta=i_beta,
        u_alpha=u_alpha, u_beta=u_beta,
        p=p, q=q,
        iAp=iAp, iBp=iBp, iCp=iCp,
        iAq=iAq, iBq=iBq, iCq=iCq,
        i_alpha_p=i_alpha_p, i_beta_p=i_beta_p,
        i_alpha_q=i_alpha_q, i_beta_q=i_beta_q,
        Is_p_rms=Is_p_rms, Is_q_rms=Is_q_rms,
        cos_phi=cos_phi,
    )


def run_simulation(label, locked_rotor=False, t_end_override=None,
                   Ts_prof=None, Tr_prof=None):
    """Запуск интегрирования и постобработка. Возвращает словарь результатов."""
    global LOCKED_ROTOR
    LOCKED_ROTOR = locked_rotor
    sim_t_end = t_end_override if t_end_override is not None else t_end


    Ts_p = Ts_prof if Ts_prof is not None else Ts_profile
    Tr_p = Tr_prof if Tr_prof is not None else Tr_profile
    _Ts_func[0] = _build_temp_interp(Ts_p, sim_t_end)
    _Tr_func[0] = _build_temp_interp(Tr_p, sim_t_end)

    Ts_start = Ts_p[0]; Ts_end = Ts_p[-1]
    Tr_start = Tr_p[0]; Tr_end = Tr_p[-1]

    lr_str = "  [ЗАТОРМОЖЕННЫЙ РОТОР]" if locked_rotor else ""
    _header(f"АД ADM174 -- ЛИНЕЙНАЯ МОДЕЛЬ (Lm = const){lr_str}\n"
            f"  Схема соединения: {CONN_MODE}\n"
            f"  Температура: Ts = {Ts_p} °C,  Tr = {Tr_p} °C")
    print(f"  Lm_3ph = {Lm_nom*1e3:.3f} мГн = const")
    print(f"  Lm_dq  = {Lm_dq_nom*1e3:.3f} мГн = const")

    print(f"\n  Интегрирование (t_end = {sim_t_end} с)...", flush=True)
    t0 = timer.time()
    y0 = np.zeros(8)
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
    Me_ = np.zeros(Nt); Im_a = np.zeros(Nt)
    is0_ = np.zeros(Nt)
    i_neut = np.zeros(Nt)
    us0_ = np.zeros(Nt)
    u_neut = np.zeros(Nt)
    Rs_arr = np.zeros(Nt)
    Rr_arr = np.zeros(Nt)

    print(f"  Постобработка ({Nt} точек)...", flush=True)
    t0p = timer.time()
    for k in range(Nt):
        if CONN_MODE == 'star_isolated':
            i6 = flux_to_currents_isolated(sol.y[:6, k], theta_e[k])
        else:
            i6 = flux_to_currents(sol.y[:6, k], theta_e[k])

        iA_[k]=i6[0]; iB_[k]=i6[1]; iC_[k]=i6[2]
        ia_[k]=i6[3]; ib_[k]=i6[4]; ic_[k]=i6[5]

        cos_sr = np.cos(theta_e[k] + _SR_ANGLES)
        Im_a[k] = _compute_Im_dq(i6, cos_sr)
        Me_[k] = electromagnetic_torque(i6, theta_e[k])

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
            u_n = compute_neutral_voltage(sol.y[:6, k], i6, u6, theta_e[k], R_ph_k)
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


    uA_ = np.zeros(Nt); uB_ = np.zeros(Nt); uC_ = np.zeros(Nt)
    for k in range(Nt):
        uA_[k], uB_[k], uC_[k] = supply_voltages_phase(t_arr[k])
    pq = pq_decomposition(iA_, iB_, iC_, uA_, uB_, uC_)

    idx_pre = np.searchsorted(t_arr, min(1.49, t_arr[-1] - 0.01))
    _subheader(f"РЕЗУЛЬТАТЫ ({label}, {CONN_MODE})\n"
               f"  Температура: Ts {Ts_start:.0f}->{Ts_end:.0f} °C,  "
               f"Tr {Tr_start:.0f}->{Tr_end:.0f} °C\n"
               f"  Rs: {Rs_arr[0]*1e3:.3f} -> {Rs_arr[-1]*1e3:.3f} мОм,  "
               f"Rr: {Rr_arr[0]*1e3:.3f} -> {Rr_arr[-1]*1e3:.3f} мОм\n"
               f"  Lm_3ph = {Lm_nom*1e3:.3f} мГн = const")

    if locked_rotor:
        print(f"\n  > ЗАТОРМОЖЕННЫЙ РОТОР (установившийся, t = {t_arr[-1]:.2f} с):"
              f"\n      n = 0 об/мин,  s = 1.0"
              f"\n      Me (уст.) = {Me_[-1]:.2f} Н*м"
              f"\n      Is (уст.) = {Is_rms_[-1]:.1f} А"
              f"\n      Ps (уст.) = {Ps_[-1]/1e3:.1f} кВт"
              f"\n      Im_dq = {Im_a[-1]:.1f} А"
              f"\n      i_n (уст.) = {i_neut[-1]:.3f} А")
        print(f"\n  > Переходный процесс:"
              f"\n      Max|Me| = {np.max(np.abs(Me_)):.0f} Н*м"
              f"\n      Max|iA| = {np.max(np.abs(iA_)):.0f} А"
              f"\n      Max|Is| = {np.max(Is_rms_):.0f} А"
              f"\n      Im max  = {np.max(Im_a):.0f} А"
              f"\n      Max|i_n| = {np.max(np.abs(i_neut)):.3f} А")
    else:
        print(f"\n  > ХХ (t ~= 1.49 с):"
              f"\n      n = {n_rpm_arr[idx_pre]:.1f} об/мин,  s = {slip_arr[idx_pre]:.6f}"
              f"\n      Me = {Me_[idx_pre]:.2f} Н*м"
              f"\n      Is = {Is_rms_[idx_pre]:.1f} А  (паспорт Io = 57.45 А)"
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
              f"\n      Im max  = {np.max(Im_a):.0f} А"
              f"\n      Max|i_n| = {np.max(np.abs(i_neut)):.3f} А")

    # --- Разложение Акаги: печать ---
    print(f"\n  > Разложение тока (Акаги p-q):")
    if locked_rotor:
        idx_pq = -1
        pq_label = "уст."
    else:
        idx_pq = idx_pre
        pq_label = "ХХ"
    print(f"      [{pq_label}] Is_p (акт.) = {pq['Is_p_rms'][idx_pq]:.1f} А")
    print(f"      [{pq_label}] Is_q (реакт.) = {pq['Is_q_rms'][idx_pq]:.1f} А")
    print(f"      [{pq_label}] cos φ = {pq['cos_phi'][idx_pq]:.4f}")
    print(f"      [{pq_label}] p = {pq['p'][idx_pq]/1e3:.1f} кВт,  q = {pq['q'][idx_pq]/1e3:.1f} квар")
    print(f"      [{pq_label}] Im_dq = {Im_a[idx_pq]:.1f} А  (для сравнения с Is_q)")
    if not locked_rotor:
        print(f"      [Нагр] Is_p (акт.) = {pq['Is_p_rms'][-1]:.1f} А")
        print(f"      [Нагр] Is_q (реакт.) = {pq['Is_q_rms'][-1]:.1f} А")
        print(f"      [Нагр] cos φ = {pq['cos_phi'][-1]:.4f}")
        print(f"      [Нагр] p = {pq['p'][-1]/1e3:.1f} кВт,  q = {pq['q'][-1]/1e3:.1f} квар")
        print(f"      [Нагр] Im_dq = {Im_a[-1]:.1f} А  (для сравнения с Is_q)")

    return dict(
        t=t_arr, sol=sol,
        PsiA=sol.y[0], PsiB=sol.y[1], PsiC=sol.y[2],
        Psi_a=sol.y[3], Psi_b=sol.y[4], Psi_c=sol.y[5],
        omega=omega_m, theta=theta_e,
        n_rpm=n_rpm_arr, slip=slip_arr,
        iA=iA_, iB=iB_, iC=iC_,
        ia=ia_, ib=ib_, ic=ic_,
        Me=Me_, Im=Im_a,
        is0=is0_, i_neutral=i_neut,
        us0=us0_, u_neutral=u_neut,
        Is_rms=Is_rms_, Ps=Ps_, Pmech=Pmech_,
        uA=uA_, uB=uB_, uC=uC_,
        pq=pq,
        Rs=Rs_arr, Rr=Rr_arr,
        label=label,
    )



print(f"\n  Параметры линейной модели:")
print(f"    Lm_3ph = {Lm_nom*1e3:.3f} мГн = const")
print(f"    Lm_dq  = {Lm_dq_nom*1e3:.3f} мГн = const")
print(f"    Lls    = {Lls*1e3:.3f} мГн")
print(f"    Llr    = {Llr*1e3:.3f} мГн")
print(f"    Rs_20  = {Rs_20*1e3:.3f} мОм")
print(f"    Rr_20  = {Rr_20*1e3:.3f} мОм")
print(f"    J      = {J} кг*м^2")
print(f"    p      = {p_poles}")
print(f"    Un     = {Un_line} В (линейное)")
print(f"    f      = {f} Гц")
print(f"    sigma  = {sigma_nom:.6f}")


res = run_simulation(label="Линейная модель")


t_end_lr = 0.5
res_lr = run_simulation(label="Заторм. ротор (линейн)",
                        locked_rotor=True, t_end_override=t_end_lr)

LOCKED_ROTOR = False


_header("РЕЗУЛЬТАТЫ ЛИНЕЙНОЙ МОДЕЛИ")

idx_s = np.searchsorted(res['t'], 1.49)

_section("ХХ (t ~= 1.49 с)")
print(f"    n = {res['n_rpm'][idx_s]:.1f} об/мин"
      f"\n    s = {res['slip'][idx_s]:.6f}"
      f"\n    Me = {res['Me'][idx_s]:.2f} Н*м"
      f"\n    Is = {res['Is_rms'][idx_s]:.1f} А"
      f"\n    Im_dq = {res['Im'][idx_s]:.1f} А"
      f"\n    i_n = {res['i_neutral'][idx_s]:.3f} А")

_section("Нагрузка (конец)")
print(f"    n = {res['n_rpm'][-1]:.1f} об/мин"
      f"\n    s = {res['slip'][-1]:.6f}"
      f"\n    Me = {res['Me'][-1]:.2f} Н*м"
      f"\n    Is = {res['Is_rms'][-1]:.1f} А"
      f"\n    Ps = {res['Ps'][-1]/1e3:.1f} кВт"
      f"\n    Pmech = {res['Pmech'][-1]/1e3:.1f} кВт"
      f"\n    i_n = {res['i_neutral'][-1]:.3f} А")

_section("Пуск")
print(f"    Max|Me| = {np.max(np.abs(res['Me'])):.0f} Н*м"
      f"\n    Max|iA| = {np.max(np.abs(res['iA'])):.0f} А"
      f"\n    Max|i_n| = {np.max(np.abs(res['i_neutral'])):.3f} А")

_section("Заторможенный ротор (установившийся)")
print(f"    Me = {res_lr['Me'][-1]:.2f} Н*м"
      f"\n    Is = {res_lr['Is_rms'][-1]:.1f} А"
      f"\n    Ps = {res_lr['Ps'][-1]/1e3:.1f} кВт"
      f"\n    i_n = {res_lr['i_neutral'][-1]:.3f} А")


t_arr = res['t']
t_ms = t_arr * 1000
iA = res['iA']; iB = res['iB']; iC = res['iC']
ia = res['ia']; ib = res['ib']; ic = res['ic']
Me = res['Me']; Im_arr = res['Im']
n_rpm = res['n_rpm']; slip = res['slip']
PsiA = res['PsiA']; PsiB = res['PsiB']; PsiC = res['PsiC']
Psi_a = res['Psi_a']; Psi_b = res['Psi_b']; Psi_c = res['Psi_c']
Is_rms = res['Is_rms']; Ps = res['Ps']; Pmech = res['Pmech']
is0 = res['is0']; i_neutral = res['i_neutral']
us0 = res['us0']; u_neutral = res['u_neutral']

plt.rcParams.update({
    'font.size': 9, 'figure.dpi': 150, 'lines.linewidth': 0.5,
    'axes.grid': True, 'grid.alpha': 0.3,
})
cs = ['#d62728', '#1f77b4', '#2ca02c']
t_ld = 1500

def vline(ax): ax.axvline(t_ld, c='red', lw=0.5, ls=':', alpha=0.5)


fig = plt.figure(figsize=(17, 26))
fig.suptitle(
    f'АД ADM174 -- линейная модель (Lm = const, {CONN_MODE})\n'
    f'Lm_3ph = {Lm_nom*1e3:.3f} мГн,  Lm_dq = {Lm_dq_nom*1e3:.3f} мГн',
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
ax.plot(t_ms,Im_arr,c='#17becf',lw=0.5); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_m$ (dq), А'); ax.set_title('Ток намагничивания')

ax=fig.add_subplot(gs[4,1])
ax.plot(t_ms,Is_rms,c='#17becf',lw=0.4)
ax.axhline(57.45,c='green',ls=':',lw=0.8,label='$I_0$=57.45')
ax.axhline(204.42,c='gray',ls=':',lw=0.8,label='$I_n$=204.42'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_s$, А'); ax.set_title('Действ. ток'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[5,0])
ax.plot(t_ms,Ps/1e3,c='#e377c2',lw=0.3,label='$P_{эл}$')
ax.plot(t_ms,Pmech/1e3,c='#bcbd22',lw=0.3,label='$P_{мех}$')
ax.axhline(170,c='gray',ls=':',lw=0.8,label='$P_{2n}$=170'); vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('P, кВт'); ax.set_title('Мощности'); ax.legend(fontsize=7)

ax=fig.add_subplot(gs[5,1])
ax.plot(t_ms, is0, c='#ff6600', lw=0.5, label='$i_s^{(0)}$')
ax.plot(t_ms, i_neutral, c='#9467bd', lw=0.3, ls='--', label='$i_n = \\sqrt{3} \\cdot i_s^{(0)}$')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('А')
ax.set_title(f'Нулевая составляющая тока ({CONN_MODE})')
ax.legend(fontsize=7)

ax=fig.add_subplot(gs[6,0])
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

ax=fig.add_subplot(gs[6,1])
ax.plot(t_ms, res['Rs']*1e3, c='#d62728', lw=0.7, label='$R_s$')
ax.plot(t_ms, res['Rr']*1e3, c='#1f77b4', lw=0.7, label='$R_r$')
ax.set_xlabel('t, мс'); ax.set_ylabel('R, мОм')
ax.set_title('Сопротивления (температурная зависимость)')
ax.legend(fontsize=7)

plt.savefig('adm174_linear_full.png', dpi=150, bbox_inches='tight')
_fig_saved("1 (полная картина)")

pq = res['pq']
T_period = 1.0 / f

fig_pq = plt.figure(figsize=(17, 22))
fig_pq.suptitle(
    f'Разложение тока на активную и реактивную составляющие (Акаги p-q)\n'
    f'АД ADM174, линейная модель, {CONN_MODE}',
    fontsize=12, fontweight='bold', y=0.995)
gs_pq = GridSpec(5, 2, hspace=0.45, wspace=0.25,
                 left=0.06, right=0.97, top=0.96, bottom=0.03)


ax = fig_pq.add_subplot(gs_pq[0, 0])
ax.plot(t_ms, pq['p']/1e3, c='#d62728', lw=0.3, label='$p(t)$')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('кВт')
ax.set_title('Мгновенная активная мощность $p(t)$'); ax.legend(fontsize=7)

ax = fig_pq.add_subplot(gs_pq[0, 1])
ax.plot(t_ms, pq['q']/1e3, c='#1f77b4', lw=0.3, label='$q(t)$')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('квар')
ax.set_title('Мгновенная реактивная мощность $q(t)$'); ax.legend(fontsize=7)


ax = fig_pq.add_subplot(gs_pq[1, 0])
ax.plot(t_ms, pq['Is_p_rms'], c='#d62728', lw=0.4, label='$I_s^{(p)}$ акт.')
ax.plot(t_ms, pq['Is_q_rms'], c='#1f77b4', lw=0.4, label='$I_s^{(q)}$ реакт.')
ax.plot(t_ms, Is_rms, c='gray', lw=0.3, ls='--', label='$I_s$ полн.')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('А')
ax.set_title('Действ. значения составляющих тока'); ax.legend(fontsize=7)


ax = fig_pq.add_subplot(gs_pq[1, 1])
ax.plot(t_ms, pq['cos_phi'], c='#2ca02c', lw=0.4)
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('cos phi')
ax.set_title('Коэффициент мощности cos phi'); ax.set_ylim(-0.05, 1.05)


mask_xx_pq = (t_arr >= 1.42) & (t_arr < 1.5)
ax = fig_pq.add_subplot(gs_pq[2, 0])
ax.plot(t_arr[mask_xx_pq]*1e3, iA[mask_xx_pq], c='k', lw=0.8, label='$i_A$ полный')
ax.plot(t_arr[mask_xx_pq]*1e3, pq['iAp'][mask_xx_pq], c='#d62728', lw=0.8, label='$i_A^{(p)}$ акт.')
ax.plot(t_arr[mask_xx_pq]*1e3, pq['iAq'][mask_xx_pq], c='#1f77b4', lw=0.8, label='$i_A^{(q)}$ реакт.')
ax.set_xlabel('t, мс'); ax.set_ylabel('А')
ax.set_title('ХХ: разложение $i_A$ (4 периода)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)


mask_ld_pq = t_arr >= (t_arr[-1] - 4*T_period)
ax = fig_pq.add_subplot(gs_pq[2, 1])
ax.plot(t_arr[mask_ld_pq]*1e3, iA[mask_ld_pq], c='k', lw=0.8, label='$i_A$ полный')
ax.plot(t_arr[mask_ld_pq]*1e3, pq['iAp'][mask_ld_pq], c='#d62728', lw=0.8, label='$i_A^{(p)}$ акт.')
ax.plot(t_arr[mask_ld_pq]*1e3, pq['iAq'][mask_ld_pq], c='#1f77b4', lw=0.8, label='$i_A^{(q)}$ реакт.')
ax.set_xlabel('t, мс'); ax.set_ylabel('А')
ax.set_title('Нагрузка: разложение $i_A$ (4 периода)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)


ax = fig_pq.add_subplot(gs_pq[3, 0])
uA_xx = res['uA'][mask_xx_pq]
iAp_xx = pq['iAp'][mask_xx_pq]
iAq_xx = pq['iAq'][mask_xx_pq]

u_norm = uA_xx / np.max(np.abs(uA_xx)) if np.max(np.abs(uA_xx)) > 0 else uA_xx
ip_norm = iAp_xx / np.max(np.abs(iA[mask_xx_pq])) if np.max(np.abs(iA[mask_xx_pq])) > 0 else iAp_xx
iq_norm = iAq_xx / np.max(np.abs(iA[mask_xx_pq])) if np.max(np.abs(iA[mask_xx_pq])) > 0 else iAq_xx
ax.plot(t_arr[mask_xx_pq]*1e3, u_norm, c='#ff7f0e', lw=1.0, label='$u_A$ (норм.)')
ax.plot(t_arr[mask_xx_pq]*1e3, ip_norm, c='#d62728', lw=0.8, ls='--', label='$i_A^{(p)}$ (норм.)')
ax.plot(t_arr[mask_xx_pq]*1e3, iq_norm, c='#1f77b4', lw=0.8, ls='--', label='$i_A^{(q)}$ (норм.)')
ax.set_xlabel('t, мс'); ax.set_ylabel('отн. ед.')
ax.set_title('ХХ: фазовые соотношения $u_A$, $i_A^{(p)}$, $i_A^{(q)}$')
ax.legend(fontsize=7); ax.grid(alpha=0.3)


ax = fig_pq.add_subplot(gs_pq[3, 1])
uA_ld = res['uA'][mask_ld_pq]
iAp_ld = pq['iAp'][mask_ld_pq]
iAq_ld = pq['iAq'][mask_ld_pq]
u_norm = uA_ld / np.max(np.abs(uA_ld)) if np.max(np.abs(uA_ld)) > 0 else uA_ld
ip_norm = iAp_ld / np.max(np.abs(iA[mask_ld_pq])) if np.max(np.abs(iA[mask_ld_pq])) > 0 else iAp_ld
iq_norm = iAq_ld / np.max(np.abs(iA[mask_ld_pq])) if np.max(np.abs(iA[mask_ld_pq])) > 0 else iAq_ld
ax.plot(t_arr[mask_ld_pq]*1e3, u_norm, c='#ff7f0e', lw=1.0, label='$u_A$ (норм.)')
ax.plot(t_arr[mask_ld_pq]*1e3, ip_norm, c='#d62728', lw=0.8, ls='--', label='$i_A^{(p)}$ (норм.)')
ax.plot(t_arr[mask_ld_pq]*1e3, iq_norm, c='#1f77b4', lw=0.8, ls='--', label='$i_A^{(q)}$ (норм.)')
ax.set_xlabel('t, мс'); ax.set_ylabel('отн. ед.')
ax.set_title('Нагрузка: фазовые соотношения $u_A$, $i_A^{(p)}$, $i_A^{(q)}$')
ax.legend(fontsize=7); ax.grid(alpha=0.3)


ax = fig_pq.add_subplot(gs_pq[4, 0])
ax.plot(t_ms, pq['Is_q_rms'], c='#1f77b4', lw=0.4, label='$I_s^{(q)}$ реакт.')
ax.plot(t_ms, Im_arr, c='#e377c2', lw=0.4, ls='--', label='$I_m$ (dq)')
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('А')
ax.set_title('Сравнение: $I_s^{(q)}$ vs $I_m$ (dq)'); ax.legend(fontsize=7)


ax = fig_pq.add_subplot(gs_pq[4, 1])
Is_check = np.sqrt(pq['Is_p_rms']**2 + pq['Is_q_rms']**2)
err_pct = np.where(Is_rms > 1e-3, (Is_check - Is_rms) / Is_rms * 100, 0.0)
ax.plot(t_ms, err_pct, c='#9467bd', lw=0.4)
vline(ax)
ax.set_xlabel('t, мс'); ax.set_ylabel('ошибка, %')
ax.set_title('Проверка: $\\sqrt{I_p^2+I_q^2}$ vs $I_s$ (отн. ошибка)')

plt.savefig('adm174_linear_pq_decomposition.png', dpi=150, bbox_inches='tight')
_fig_saved("PQ (разложение Акаги)")


fig_pq2, axes_pq2 = plt.subplots(2, 2, figsize=(15, 9))
fig_pq2.suptitle(f'Разложение тока при пуске (Акаги p-q, {CONN_MODE}, 0-200 мс)',
                 fontsize=12, fontweight='bold')
m_start = t_arr <= 0.2

axes_pq2[0,0].plot(t_arr[m_start]*1e3, pq['Is_p_rms'][m_start], c='#d62728', lw=0.5, label='$I_s^{(p)}$ акт.')
axes_pq2[0,0].plot(t_arr[m_start]*1e3, pq['Is_q_rms'][m_start], c='#1f77b4', lw=0.5, label='$I_s^{(q)}$ реакт.')
axes_pq2[0,0].plot(t_arr[m_start]*1e3, Is_rms[m_start], c='gray', lw=0.3, ls='--', label='$I_s$ полн.')
axes_pq2[0,0].set_title('Составляющие тока при пуске'); axes_pq2[0,0].legend(fontsize=7); axes_pq2[0,0].grid(alpha=0.3)

axes_pq2[0,1].plot(t_arr[m_start]*1e3, pq['cos_phi'][m_start], c='#2ca02c', lw=0.5)
axes_pq2[0,1].set_title('cos φ при пуске'); axes_pq2[0,1].set_ylim(-0.05, 1.05); axes_pq2[0,1].grid(alpha=0.3)

axes_pq2[1,0].plot(t_arr[m_start]*1e3, pq['p'][m_start]/1e3, c='#d62728', lw=0.4, label='$p(t)$')
axes_pq2[1,0].set_title('Акт. мощность при пуске'); axes_pq2[1,0].set_ylabel('кВт')
axes_pq2[1,0].legend(fontsize=7); axes_pq2[1,0].grid(alpha=0.3)

axes_pq2[1,1].plot(t_arr[m_start]*1e3, pq['q'][m_start]/1e3, c='#1f77b4', lw=0.4, label='$q(t)$')
axes_pq2[1,1].set_title('Реакт. мощность при пуске'); axes_pq2[1,1].set_ylabel('квар')
axes_pq2[1,1].legend(fontsize=7); axes_pq2[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('adm174_linear_pq_startup.png', dpi=150, bbox_inches='tight')
_fig_saved("PQ-2 (пуск)")

n_show = 3  # число периодов
t_xx_start = 1.5 - n_show * T_period
mask_xx3 = (t_arr >= t_xx_start) & (t_arr < 1.5)
t_plot = t_arr[mask_xx3] * 1e3

uA_plot = res['uA'][mask_xx3]
iA_plot = iA[mask_xx3]
iAp_plot = pq['iAp'][mask_xx3]
iAq_plot = pq['iAq'][mask_xx3]

fig_xx, ax1 = plt.subplots(figsize=(16, 7))
fig_xx.suptitle(
    f'ХХ, установившийся режим, фаза A -- {n_show} периода ({CONN_MODE})\n'
    f'Разложение тока на активную и реактивную составляющие (Акаги)',
    fontsize=12, fontweight='bold')


color_u = '#ff7f0e'
ax1.plot(t_plot, uA_plot, c=color_u, lw=1.2, label='$u_A$')
ax1.set_xlabel('t, мс', fontsize=11)
ax1.set_ylabel('$u_A$, В', color=color_u, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_u)


ax2 = ax1.twinx()
ax2.plot(t_plot, iA_plot, c='k', lw=1.2, label='$i_A$ полный')
ax2.plot(t_plot, iAp_plot, c='#d62728', lw=1.2, ls='--', label='$i_A^{(p)}$ активный')
ax2.plot(t_plot, iAq_plot, c='#1f77b4', lw=1.2, ls='--', label='$i_A^{(q)}$ реактивный')
ax2.set_ylabel('i, А', fontsize=11)


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('adm174_linear_pq_xx_phaseA.png', dpi=150, bbox_inches='tight')
_fig_saved("PQ-3 (ХХ фаза A)")

n_show_ld = 3

t_ld_end = t_arr[-1] - 2 * T_period
t_ld_start = t_ld_end - n_show_ld * T_period
mask_ld3 = (t_arr >= t_ld_start) & (t_arr <= t_ld_end)
t_plot_ld = t_arr[mask_ld3] * 1e3

fig_ld, ax1 = plt.subplots(figsize=(16, 7))
fig_ld.suptitle(
    f'Нагрузка 1096 Н·м, установившийся режим, фаза A -- {n_show_ld} периода ({CONN_MODE})\n'
    f'Разложение тока на активную и реактивную составляющие (Акаги)',
    fontsize=12, fontweight='bold')

color_u = '#ff7f0e'
ax1.plot(t_plot_ld, res['uA'][mask_ld3], c=color_u, lw=1.2, label='$u_A$')
ax1.set_xlabel('t, мс', fontsize=11)
ax1.set_ylabel('$u_A$, В', color=color_u, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_u)

ax2 = ax1.twinx()
ax2.plot(t_plot_ld, iA[mask_ld3], c='k', lw=1.2, label='$i_A$ полный')
ax2.plot(t_plot_ld, pq['iAp'][mask_ld3], c='#d62728', lw=1.2, ls='--', label='$i_A^{(p)}$ активный')
ax2.plot(t_plot_ld, pq['iAq'][mask_ld3], c='#1f77b4', lw=1.2, ls='--', label='$i_A^{(q)}$ реактивный')
ax2.set_ylabel('i, А', fontsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('adm174_linear_pq_load_phaseA.png', dpi=150, bbox_inches='tight')
_fig_saved("PQ-4 (Нагрузка фаза A)")

pq_lr = res_lr['pq']
t_lr = res_lr['t']
T_lr = 1.0 / f
n_show_lr = 3
t_lr_end = t_lr[-1] - 2 * T_lr
t_lr_start = t_lr_end - n_show_lr * T_lr
mask_lr3 = (t_lr >= t_lr_start) & (t_lr <= t_lr_end)
t_plot_lr = t_lr[mask_lr3] * 1e3

fig_lr_pq, ax1 = plt.subplots(figsize=(16, 7))
fig_lr_pq.suptitle(
    f'Заторможенный ротор, установившийся режим, фаза A -- {n_show_lr} периода ({CONN_MODE})\n'
    f'Разложение тока на активную и реактивную составляющие (Акаги)',
    fontsize=12, fontweight='bold')

color_u = '#ff7f0e'
ax1.plot(t_plot_lr, res_lr['uA'][mask_lr3], c=color_u, lw=1.2, label='$u_A$')
ax1.set_xlabel('t, мс', fontsize=11)
ax1.set_ylabel('$u_A$, В', color=color_u, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_u)

ax2 = ax1.twinx()
ax2.plot(t_plot_lr, res_lr['iA'][mask_lr3], c='k', lw=1.2, label='$i_A$ полный')
ax2.plot(t_plot_lr, pq_lr['iAp'][mask_lr3], c='#d62728', lw=1.2, ls='--', label='$i_A^{(p)}$ активный')
ax2.plot(t_plot_lr, pq_lr['iAq'][mask_lr3], c='#1f77b4', lw=1.2, ls='--', label='$i_A^{(q)}$ реактивный')
ax2.set_ylabel('i, А', fontsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('adm174_linear_pq_lr_phaseA.png', dpi=150, bbox_inches='tight')
_fig_saved("PQ-5 (Заторм. ротор фаза A)")

fig3, axes = plt.subplots(2, 2, figsize=(15, 9))
fig3.suptitle(f'Пуск -- линейная модель ({CONN_MODE}, 0-200 мс)', fontsize=12, fontweight='bold')
m = t_arr <= 0.2
axes[0,0].plot(t_arr[m]*1e3,iA[m],c=cs[0],lw=0.5,label='$i_A$')
axes[0,0].plot(t_arr[m]*1e3,iB[m],c=cs[1],lw=0.5,label='$i_B$')
axes[0,0].plot(t_arr[m]*1e3,iC[m],c=cs[2],lw=0.5,label='$i_C$')
axes[0,0].set_title('Пуск. токи'); axes[0,0].legend(fontsize=7); axes[0,0].grid(alpha=0.3)
axes[0,1].plot(t_arr[m]*1e3,Im_arr[m],c='#17becf',lw=0.7)
axes[0,1].set_title('Im при пуске'); axes[0,1].grid(alpha=0.3)
axes[1,0].plot(t_arr[m]*1e3,Me[m],c='#9467bd',lw=0.4); axes[1,0].set_title('Пуск. момент'); axes[1,0].grid(alpha=0.3)
axes[1,1].plot(t_arr[m]*1e3,i_neutral[m],c='#ff6600',lw=0.5)
axes[1,1].set_title(f'Ток нейтрали при пуске ({CONN_MODE})'); axes[1,1].set_ylabel('$i_n$, А'); axes[1,1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('adm174_linear_startup.png', dpi=150, bbox_inches='tight')
_fig_saved("2 (пуск)")

fig4, axes4 = plt.subplots(2, 2, figsize=(15, 9))
fig4.suptitle(f'Наброс 1096 Н*м -- линейная модель ({CONN_MODE})', fontsize=12, fontweight='bold')
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
plt.savefig('adm174_linear_load.png', dpi=150, bbox_inches='tight')
_fig_saved("3 (наброс нагрузки)")


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


# ХХ
t_xx_end = 1.5
mask_xx = t_arr < t_xx_end
t_xx = t_arr[mask_xx]

t_w_xxA, iA_w_xx, harm_xxA, thd_xxA = harmonic_analysis(t_xx, iA[mask_xx], f, 4)
t_w_xxB, iB_w_xx, harm_xxB, thd_xxB = harmonic_analysis(t_xx, iB[mask_xx], f, 4)
t_w_xxC, iC_w_xx, harm_xxC, thd_xxC = harmonic_analysis(t_xx, iC[mask_xx], f, 4)
t_w_xx0, in_w_xx, harm_xx0, thd_xx0 = harmonic_analysis(t_xx, i_neutral[mask_xx], f, 4)

# Нагрузка
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
ax.set_title(f'ХХ -- спектр $i_n$')
ax.set_xlim(0, 25); ax.grid(alpha=0.3, axis='y')

plt.savefig('adm174_linear_harmonics_xx.png', dpi=150, bbox_inches='tight')
_fig_saved("4 (гармоники ХХ)")

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

plt.savefig('adm174_linear_harmonics_load.png', dpi=150, bbox_inches='tight')
_fig_saved("5 (гармоники Нагрузка)")

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
plt.savefig('adm174_linear_zero_seq.png', dpi=150, bbox_inches='tight')
_fig_saved("6 (нулевые составляющие)")

_header("ЗАТОРМОЖЕННЫЙ РОТОР (линейная модель)")

t_lr = res_lr['t']
t_ms_lr = t_lr * 1000

fig_lr = plt.figure(figsize=(17, 18))
fig_lr.suptitle(
    f'ЗАТОРМОЖЕННЫЙ РОТОР -- линейная модель ({CONN_MODE})',
    fontsize=13, fontweight='bold', y=0.995)
gs_lr = GridSpec(4, 2, hspace=0.45, wspace=0.25,
                 left=0.06, right=0.97, top=0.96, bottom=0.03)

ax = fig_lr.add_subplot(gs_lr[0, 0])
ax.plot(t_ms_lr, res_lr['iA'], c=cs[0], lw=0.5, label='$i_A$')
ax.plot(t_ms_lr, res_lr['iB'], c=cs[1], lw=0.5, label='$i_B$')
ax.plot(t_ms_lr, res_lr['iC'], c=cs[2], lw=0.5, label='$i_C$')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Токи статора'); ax.legend(fontsize=7)

ax = fig_lr.add_subplot(gs_lr[0, 1])
ax.plot(t_ms_lr, res_lr['ia'], c=cs[0], lw=0.5, label='$i_a$')
ax.plot(t_ms_lr, res_lr['ib'], c=cs[1], lw=0.5, label='$i_b$')
ax.plot(t_ms_lr, res_lr['ic'], c=cs[2], lw=0.5, label='$i_c$')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Токи ротора'); ax.legend(fontsize=7)

ax = fig_lr.add_subplot(gs_lr[1, 0])
ax.plot(t_ms_lr, res_lr['Me'], c='#9467bd', lw=0.5)
ax.axhline(0, c='k', lw=0.3)
ax.set_xlabel('t, мс'); ax.set_ylabel('$M_e$, Н*м')
ax.set_title('Электромагнитный момент')

ax = fig_lr.add_subplot(gs_lr[1, 1])
ax.plot(t_ms_lr, res_lr['PsiA'], c=cs[0], lw=0.5, label='$\\Psi_A$')
ax.plot(t_ms_lr, res_lr['PsiB'], c=cs[1], lw=0.5, label='$\\Psi_B$')
ax.plot(t_ms_lr, res_lr['PsiC'], c=cs[2], lw=0.5, label='$\\Psi_C$')
ax.set_xlabel('t, мс'); ax.set_ylabel('Psi, Вб')
ax.set_title('Потокосцепление статора'); ax.legend(fontsize=7)

ax = fig_lr.add_subplot(gs_lr[2, 0])
ax.plot(t_ms_lr, res_lr['Is_rms'], c='#17becf', lw=0.5)
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_s$, А')
ax.set_title('Действующий ток статора')

ax = fig_lr.add_subplot(gs_lr[2, 1])
ax.plot(t_ms_lr, res_lr['Ps']/1e3, c='#e377c2', lw=0.5)
ax.set_xlabel('t, мс'); ax.set_ylabel('P, кВт')
ax.set_title('Электрическая мощность (Pmech=0)')

ax = fig_lr.add_subplot(gs_lr[3, 0])
ax.plot(t_ms_lr, res_lr['i_neutral'], c='#ff6600', lw=0.5)
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title(f'Ток нейтрали ({CONN_MODE})')

ax = fig_lr.add_subplot(gs_lr[3, 1])
ax.plot(t_ms_lr, res_lr['Im'], c='#17becf', lw=0.5)
ax.set_xlabel('t, мс'); ax.set_ylabel('$I_m$ (dq), А')
ax.set_title('Ток намагничивания')

plt.savefig('adm174_linear_locked_rotor.png', dpi=150, bbox_inches='tight')
_fig_saved("7 (заторможенный ротор)")

fig_lr2, axes_lr2 = plt.subplots(2, 2, figsize=(15, 9))
fig_lr2.suptitle(f'Заторможенный ротор -- установившийся режим ({CONN_MODE})',
                 fontsize=12, fontweight='bold')

T_lr = 1.0 / f
mask_lr_ss = t_lr >= (t_lr[-1] - 4*T_lr)

ax = axes_lr2[0, 0]
ax.plot(t_lr[mask_lr_ss]*1e3, res_lr['iA'][mask_lr_ss], c=cs[0], lw=0.8, label='$i_A$')
ax.plot(t_lr[mask_lr_ss]*1e3, res_lr['iB'][mask_lr_ss], c=cs[1], lw=0.8, label='$i_B$')
ax.plot(t_lr[mask_lr_ss]*1e3, res_lr['iC'][mask_lr_ss], c=cs[2], lw=0.8, label='$i_C$')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Токи статора (уст.)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes_lr2[0, 1]
ax.plot(t_lr[mask_lr_ss]*1e3, res_lr['ia'][mask_lr_ss], c=cs[0], lw=0.8, label='$i_a$')
ax.plot(t_lr[mask_lr_ss]*1e3, res_lr['ib'][mask_lr_ss], c=cs[1], lw=0.8, label='$i_b$')
ax.plot(t_lr[mask_lr_ss]*1e3, res_lr['ic'][mask_lr_ss], c=cs[2], lw=0.8, label='$i_c$')
ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
ax.set_title('Токи ротора (уст.)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes_lr2[1, 0]
ax.plot(t_lr[mask_lr_ss]*1e3, res_lr['Me'][mask_lr_ss], c='#9467bd', lw=0.8)
ax.axhline(0, c='k', lw=0.3)
ax.set_xlabel('t, мс'); ax.set_ylabel('$M_e$, Н*м')
ax.set_title('Момент (уст.)'); ax.grid(alpha=0.3)

ax = axes_lr2[1, 1]
ax.plot(t_lr[mask_lr_ss]*1e3, res_lr['i_neutral'][mask_lr_ss], c='#ff6600', lw=0.8)
ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
ax.set_title(f'Ток нейтрали (уст., {CONN_MODE})'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('adm174_linear_locked_rotor_steady.png', dpi=150, bbox_inches='tight')
_fig_saved("8 (заторм. ротор уст. режим)")

_subheader(f"ГАРМОНИЧЕСКИЙ АНАЛИЗ -- ЗАТОРМОЖЕННЫЙ РОТОР ({CONN_MODE})", ch='-', width=80)

t_lr_arr = res_lr['t']
_, _, harm_lrA, thd_lrA = harmonic_analysis(t_lr_arr, res_lr['iA'], f, 4)
_, _, harm_lrB, thd_lrB = harmonic_analysis(t_lr_arr, res_lr['iB'], f, 4)
_, _, harm_lrC, thd_lrC = harmonic_analysis(t_lr_arr, res_lr['iC'], f, 4)
_, _, harm_lrN, thd_lrN = harmonic_analysis(t_lr_arr, res_lr['i_neutral'], f, 4)

_section("Линейная модель (последние 4 периода)")
_harm_table_header_4col()
for k_idx in range(0, min(16, len(harm_lrA))):
    aA = harm_lrA[k_idx, 2]; aB = harm_lrB[k_idx, 2]
    aC = harm_lrC[k_idx, 2]; aN = harm_lrN[k_idx, 2]
    _harm_table_row(k_idx, harm_lrA[k_idx,1], aA, aB, aC, aN)
_thd_line(thd_lrA, thd_lrB, thd_lrC)
print(f"  THD i_n = {thd_lrN:.2f}%")

_header("Готово.")