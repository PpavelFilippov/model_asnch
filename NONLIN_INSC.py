"""
Модификация модели АД ADM174 с межвитковым коротким замыканием (МВКЗ).

Базовая модель: 6 уравнений потокосцеплений (3 статор + 3 ротор) + 2 механических.
Модификация: добавлен 7-й контур -- контур замкнутых накоротко витков в одной из фаз статора.

Физика МВКЗ:
  - В фазе (по умолчанию A) N_f витков из N_s замкнуты накоротко.
  - Отношение mu = N_f / N_s определяет долю витков в КЗ.
  - КЗ-контур имеет собственное сопротивление R_f = mu * Rs (пропорционально числу витков).
  - КЗ-контур связан с основной обмоткой через взаимную индуктивность.
  - Напряжение источника прикладывается к (1 - mu) * N_s здоровым виткам фазы.

Расширенная матрица индуктивностей 7x7:
  [L_6x6  |  L_6x1_coupling ]   [ i_6   ]   [ Psi_6   ]
  [L_1x6  |  L_ff           ] * [ i_f   ] = [ Psi_f   ]

Результат: несимметрия токов → появление гармоник (особенно 3-я, 5-я) в неисправной фазе.

Кривая: Psi_m(Im) = a * arctan(b * Im)
"""

import numpy as np
from numpy.linalg import solve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time as timer


#  Параметры МВКЗ

ITSC_ENABLED = True          # Включить/выключить межвитковое КЗ
ITSC_PHASE = 0               # Фаза с МВКЗ: 0=A, 1=B, 2=C
ITSC_MU = 0.05               # Доля замкнутых витков (mu = N_f / N_s)
ITSC_R_contact = 0.001       # Ом, сопротивление контакта в месте КЗ (обычно мало)
ITSC_T_START = 0.0           # Время включения КЗ (0 = с самого начала)
ITSC_Llf = 0.1e-3            # Гн, собственная индуктивность рассеяния КЗ-контура
                              # (поток рассеяния КЗ-витков, не сцепленный с остальной обмоткой)
                              # Физически: ~10-50% от Lls. Обеспечивает невырожденность 7x7.


#  Вспомогательные функции вывода

def _header(title, ch='=', width=72):
    line = ch * width
    print(f"\n{line}\n  {title}\n{line}")

def _subheader(title, ch='-', width=60):
    line = ch * width
    print(f"\n  {line}\n  {title}\n  {line}")

def _section(title):
    print(f"\n  === {title} ===")

def _table_header(*cols, widths=None):
    if widths is None:
        widths = [30] + [10] * (len(cols) - 1)
    fmt = "    " + "  ".join(f"{{:>{w}s}}" if i else f"{{:{w}s}}"
                              for i, w in enumerate(widths))
    print(fmt.format(*cols))

def _fig_saved(name):
    print(f"  + Рис. {name} сохранён")

def _harm_table_header_4col():
    print(f"  {'Гарм':>5}  {'f, Гц':>8}  {'iA':>8}  {'iB':>8}  {'iC':>8}  {'i_n':>9}")

def _harm_table_row(k, freq, aA, aB, aC, aN):
    print(f"  {k:5d}  {freq:8.1f}  {aA:8.2f}  {aB:8.2f}  {aC:8.2f}  {aN:9.3f}")

def _thd_line(thd_A, thd_B, thd_C, prefix=""):
    print(f"  {prefix}THD iA = {thd_A:.2f}%,  THD iB = {thd_B:.2f}%,  THD iC = {thd_C:.2f}%")

def _compare_header():
    _table_header('Параметр', 'Насыщ', 'Линейн', 'Разность')


#  Параметры двигателя

CONN_MODE = 'star_grounded'
SATURATION_ENABLED = True

p_poles = 2
Rs_20 = 0.0291
Rr_20 = 0.02017
alpha_R = 0.004

Ts_profile = [20]
Tr_profile = [20]

RsA = Rs_20; RsB = Rs_20; RsC = Rs_20
Rra = Rr_20; Rrb = Rr_20; Rrc = Rr_20
R_phases = np.array([RsA, RsB, RsC, Rra, Rrb, Rrc])


def _build_temp_interp(T_profile, t_sim_end):
    T_arr = np.asarray(T_profile, dtype=float)
    n = len(T_arr)
    if n == 1:
        val = T_arr[0]
        return lambda t: val
    t_nodes = np.linspace(0.0, t_sim_end, n)
    return lambda t: np.interp(t, t_nodes, T_arr)


_Ts_func = [lambda t: 20.0]
_Tr_func = [lambda t: 20.0]


def R_phases_of_t(t):
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
sigma_nom = 1 - Lm_nom**2 / ((Lls + Lm_nom) * (Llr + Lm_nom))
t_end = 3.0


K_sat = 1.263
Lm_dq_0 = Lm_dq_nom * K_sat
Lm_0 = (2.0 / 3.0) * Lm_dq_0
Im_nom_dq = 57.45

_b_mag = 0.017019
_a_mag = Lm_dq_0 / _b_mag
_Psi_max_dq = _a_mag * np.pi / 2


def psi_m_dq(Im_dq):
    return _a_mag * np.arctan(_b_mag * Im_dq)


def Lm_dq_of_Im(Im_dq):
    bI = _b_mag * Im_dq
    if bI < 1e-12:
        return Lm_dq_0
    return Lm_dq_0 * np.arctan(bI) / bI


def Lm_3ph_of_Im(Im_dq):
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


#  Матрица индуктивностей 6x6 (базовая)

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


#  Расширенная матрица 7x7 с контуром МВКЗ

def _build_L7_itsc(theta, Lm, mu, fault_phase, Llf, out7=None):
    """
    Матрица L 7x7: базовая 6x6 + КЗ-контур.
    L_ff = mu^2*Ls + Llf, M_f_same = mu*Ls, M_f_other = -mu*Lm/2,
    M_f_rk = mu*Lm*cos(theta+phi_k).
    """
    if out7 is None:
        out7 = np.empty((7, 7))

    Ls = Lls + Lm
    Lr = Llr + Lm
    hLm = Lm * 0.5
    cos_sr = Lm * np.cos(theta + _SR_ANGLES)

    out7[0,0]=Ls;    out7[0,1]=-hLm;  out7[0,2]=-hLm
    out7[1,0]=-hLm;  out7[1,1]=Ls;    out7[1,2]=-hLm
    out7[2,0]=-hLm;  out7[2,1]=-hLm;  out7[2,2]=Ls

    out7[0,3]=cos_sr[0,0]; out7[0,4]=cos_sr[0,1]; out7[0,5]=cos_sr[0,2]
    out7[1,3]=cos_sr[1,0]; out7[1,4]=cos_sr[1,1]; out7[1,5]=cos_sr[1,2]
    out7[2,3]=cos_sr[2,0]; out7[2,4]=cos_sr[2,1]; out7[2,5]=cos_sr[2,2]

    out7[3,0]=cos_sr[0,0]; out7[3,1]=cos_sr[1,0]; out7[3,2]=cos_sr[2,0]
    out7[4,0]=cos_sr[0,1]; out7[4,1]=cos_sr[1,1]; out7[4,2]=cos_sr[2,1]
    out7[5,0]=cos_sr[0,2]; out7[5,1]=cos_sr[1,2]; out7[5,2]=cos_sr[2,2]

    out7[3,3]=Lr;    out7[3,4]=-hLm;  out7[3,5]=-hLm
    out7[4,3]=-hLm;  out7[4,4]=Lr;    out7[4,5]=-hLm
    out7[5,3]=-hLm;  out7[5,4]=-hLm;  out7[5,5]=Lr

    fp = fault_phase

    # L_ff = mu^2 * Ls + Llf  (Llf обеспечивает невырожденность)
    L_ff = mu**2 * Ls + Llf
    out7[6, 6] = L_ff

    # Взаимоиндукция с фазами статора
    for j in range(3):
        if j == fp:
            M_fj = mu * Ls
        else:
            M_fj = -mu * hLm
        out7[6, j] = M_fj
        out7[j, 6] = M_fj

    # Взаимоиндукция с фазами ротора
    for j in range(3):
        M_fr = mu * cos_sr[fp, j]
        out7[6, 3 + j] = M_fr
        out7[3 + j, 6] = M_fr

    return out7


#  Вычисление тока намагничивания

def _compute_Im_dq(i6, cos_sr):
    """Модуль тока намагничивания Im_dq (RMS)."""
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


def _compute_Im_dq_itsc(i6, i_f, cos_sr, mu, fault_phase):
    """Im_dq с учётом МВКЗ: i_eff[fp] = i[fp] - mu*i_f."""
    is_eff = i6[0:3].copy()
    is_eff[fault_phase] -= mu * i_f

    ir = i6[3:6]
    ir_s0 = cos_sr[0,0]*ir[0] + cos_sr[0,1]*ir[1] + cos_sr[0,2]*ir[2]
    ir_s1 = cos_sr[1,0]*ir[0] + cos_sr[1,1]*ir[1] + cos_sr[1,2]*ir[2]
    ir_s2 = cos_sr[2,0]*ir[0] + cos_sr[2,1]*ir[1] + cos_sr[2,2]*ir[2]

    im0 = is_eff[0] + ir_s0
    im1 = is_eff[1] + ir_s1
    im2 = is_eff[2] + ir_s2

    im_alpha = _SQRT2_OVER3 * (im0 - 0.5*im1 - 0.5*im2)
    im_beta  = _SQRT2_OVER3 * _SQRT3_2 * (im1 - im2)

    return np.sqrt(im_alpha*im_alpha + im_beta*im_beta)


def zero_seq_current(i6):
    return _INV_SQRT3 * (i6[0] + i6[1] + i6[2])


def zero_seq_voltage(uA, uB, uC):
    return _INV_SQRT3 * (uA + uB + uC)


def neutral_voltage_from_zero_seq(u_s0):
    return u_s0 / _SQRT3


#  Решение Psi -> i с насыщением

_L_buf = np.empty((6, 6))
_L_buf_7 = np.empty((7, 7))
_L_buf_7_itsc = np.empty((7, 7))
_L_buf_8 = np.empty((8, 8))


def flux_to_currents_sat(psi6, theta, Lm_prev):
    """Стандартное решение 6x6 (без МВКЗ)."""
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


def flux_to_currents_sat_itsc(psi7, theta, Lm_prev, mu, fault_phase, Llf):
    """
    Решение расширенной системы 7x7 с МВКЗ.
    psi7 = [psi_A, psi_B, psi_C, psi_a, psi_b, psi_c, psi_f]
    """
    cos_sr = np.cos(theta + _SR_ANGLES)

    if not SATURATION_ENABLED:
        Lm_fixed = Lm_nom
        _build_L7_itsc(theta, Lm_fixed, mu, fault_phase, Llf, _L_buf_7_itsc)
        i7 = solve(_L_buf_7_itsc, psi7)
        return i7[:6], i7[6], Lm_fixed

    Lm = Lm_prev

    for k in range(30):
        _build_L7_itsc(theta, Lm, mu, fault_phase, Llf, _L_buf_7_itsc)
        i7 = solve(_L_buf_7_itsc, psi7)
        i6 = i7[:6]
        i_f = i7[6]

        Im = _compute_Im_dq_itsc(i6, i_f, cos_sr, mu, fault_phase)
        Lm_new = Lm_3ph_of_Im(Im)

        err = abs(Lm_new - Lm)
        if err < 1e-9:
            return i6, i_f, Lm_new
        if err < 1e-6:
            alpha = 0.7
        elif err < 1e-4:
            alpha = 0.5
        else:
            alpha = 0.3
        Lm = alpha * Lm_new + (1.0 - alpha) * Lm

    return i6, i_f, Lm


def flux_to_currents_sat_isolated(psi6, theta, Lm_prev):
    """Решение с ограничением iA+iB+iC=0 (star_isolated), без МВКЗ."""
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


def flux_to_currents_sat_isolated_itsc(psi7, theta, Lm_prev, mu, fault_phase, Llf):
    """
    Решение 7+1 (7 потокосцеплений + ограничение iA+iB+iC=0) = 8x8 для star_isolated с МВКЗ.
    """
    cos_sr = np.cos(theta + _SR_ANGLES)
    constraint = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    if not SATURATION_ENABLED:
        Lm_fixed = Lm_nom
        _build_L7_itsc(theta, Lm_fixed, mu, fault_phase, Llf, _L_buf_7_itsc)
        _L_buf_8[:7, :7] = _L_buf_7_itsc
        _L_buf_8[:7, 7] = constraint
        _L_buf_8[7, :7] = constraint
        _L_buf_8[7, 7] = 0.0
        rhs = np.zeros(8)
        rhs[:7] = psi7
        sol8 = solve(_L_buf_8, rhs)
        return sol8[:6], sol8[6], Lm_fixed

    Lm = Lm_prev
    for k in range(30):
        _build_L7_itsc(theta, Lm, mu, fault_phase, Llf, _L_buf_7_itsc)
        _L_buf_8[:7, :7] = _L_buf_7_itsc
        _L_buf_8[:7, 7] = constraint
        _L_buf_8[7, :7] = constraint
        _L_buf_8[7, 7] = 0.0
        rhs = np.zeros(8)
        rhs[:7] = psi7
        sol8 = solve(_L_buf_8, rhs)
        i6 = sol8[:6]
        i_f = sol8[6]
        Im = _compute_Im_dq_itsc(i6, i_f, cos_sr, mu, fault_phase)
        Lm_new = Lm_3ph_of_Im(Im)
        err = abs(Lm_new - Lm)
        if err < 1e-9:
            return i6, i_f, Lm_new
        if err < 1e-6:
            alpha = 0.7
        elif err < 1e-4:
            alpha = 0.5
        else:
            alpha = 0.3
        Lm = alpha * Lm_new + (1.0 - alpha) * Lm
    return i6, i_f, Lm


#  Электромагнитный момент

def electromagnetic_torque(i6, theta, Lm, Im_dq):
    sin_sr = -Lm * np.sin(theta + _SR_ANGLES)
    is3 = i6[0:3]
    ir3 = i6[3:6]
    return p_poles * (is3 @ sin_sr @ ir3)


def electromagnetic_torque_itsc(i6, i_f, theta, Lm, mu, fault_phase):
    """Момент с учётом МВКЗ: Me = Me_main + Me_f."""
    sin_sr = -Lm * np.sin(theta + _SR_ANGLES)
    is3 = i6[0:3].copy()
    ir3 = i6[3:6]

    # Основной момент
    Me_main = p_poles * (is3 @ sin_sr @ ir3)

    # Добавка от КЗ-контура
    Me_f = -p_poles * mu * i_f * (sin_sr[fault_phase, :] @ ir3)

    return Me_main + Me_f


#  Управление моделированием

LOCKED_ROTOR = False
_last_Lm = [Lm_0]
_call_count = [0]
_wall_start = [0.0]
_last_wall = [0.0]


def Mc_func(t, omega_mech):
    return 0.0 if t < 1.5 else 1096.0


def supply_voltages_phase(t):
    uA = U_amp * np.cos(omega1 * t)
    uB = U_amp * np.cos(omega1 * t - 2*np.pi/3)
    uC = U_amp * np.cos(omega1 * t + 2*np.pi/3)
    return uA, uB, uC


def supply_voltages(t):
    uA, uB, uC = supply_voltages_phase(t)
    return np.array([uA, uB, uC, 0.0, 0.0, 0.0])


def compute_neutral_voltage(psi6, i6, u6_source, theta, Lm, R_ph=None):
    if R_ph is None:
        R_ph = R_phases
    dpsi_unconstrained = u6_source - R_ph * i6
    sum_dpsi_s = dpsi_unconstrained[0] + dpsi_unconstrained[1] + dpsi_unconstrained[2]
    u_n = sum_dpsi_s / 3.0
    return u_n


#  Система ОДУ

def itsc_active(t):
    """Проверка, активен ли МВКЗ в данный момент времени."""
    return ITSC_ENABLED and (t >= ITSC_T_START)


def ode_system(t, y):
    """
    Система ОДУ (8 без МВКЗ / 9 с МВКЗ).
    dPsi_k/dt = u_k - R_k*i_k для всех обмоток; dPsi_f/dt = -R_f*i_f.
    Связь КЗ-контура -- через L 7x7 при решении L*i=Psi.
    """
    if ITSC_ENABLED:
        psi6 = y[0:6]
        omega_mech = y[6]
        theta_elec = y[7]
        psi_f = y[8]
    else:
        psi6 = y[0:6]
        omega_mech = y[6]
        theta_elec = y[7]

    if LOCKED_ROTOR:
        omega_mech = 0.0
        theta_elec = 0.0

    R_ph = R_phases_of_t(t)
    mu = ITSC_MU
    fp = ITSC_PHASE
    Llf = ITSC_Llf

    # Решение для токов
    if itsc_active(t):
        psi7 = np.zeros(7)
        psi7[:6] = psi6
        psi7[6] = psi_f

        if CONN_MODE == 'star_isolated':
            i6, i_f, Lm_c = flux_to_currents_sat_isolated_itsc(
                psi7, theta_elec, _last_Lm[0], mu, fp, Llf)
        else:
            i6, i_f, Lm_c = flux_to_currents_sat_itsc(
                psi7, theta_elec, _last_Lm[0], mu, fp, Llf)
    else:
        i_f = 0.0
        if CONN_MODE == 'star_isolated':
            i6, Lm_c = flux_to_currents_sat_isolated(psi6, theta_elec, _last_Lm[0])
        else:
            i6, Lm_c = flux_to_currents_sat(psi6, theta_elec, _last_Lm[0])

    _last_Lm[0] = Lm_c

    u6 = supply_voltages(t)

    if CONN_MODE == 'star_isolated':
        u_n = compute_neutral_voltage(psi6, i6, u6, theta_elec, Lm_c, R_ph)
        u6[0] -= u_n
        u6[1] -= u_n
        u6[2] -= u_n

    # dPsi/dt для 6 основных обмоток
    dpsi_dt = u6 - R_ph * i6

    # Связь МВКЗ с основными обмотками -- через L 7x7, без явных mu*Rs*i_f

    # Момент
    cos_sr = np.cos(theta_elec + _SR_ANGLES)
    if itsc_active(t):
        Im = _compute_Im_dq_itsc(i6, i_f, cos_sr, mu, fp)
        Me_val = electromagnetic_torque_itsc(i6, i_f, theta_elec, Lm_c, mu, fp)
    else:
        Im = _compute_Im_dq(i6, cos_sr)
        Me_val = electromagnetic_torque(i6, theta_elec, Lm_c, Im)

    if LOCKED_ROTOR:
        domega_dt = 0.0
        dtheta_dt = 0.0
    else:
        Mc = Mc_func(t, omega_mech)
        domega_dt = (Me_val - Mc) / J
        dtheta_dt = p_poles * omega_mech

    # dPsi_f/dt = -R_f * i_f  (КЗ-контур замкнут, u_f=0)
    # ЭДС наводки -- неявно через L_7x7^{-1}: i_f != 0 при Psi_f=0
    if ITSC_ENABLED:
        if itsc_active(t):
            Rs_t = R_ph[0]
            R_f = mu * Rs_t + ITSC_R_contact
            dpsi_f_dt = -R_f * i_f
        else:
            dpsi_f_dt = 0.0

    _call_count[0] += 1
    wt = timer.time()
    if wt - _last_wall[0] > 5.0:
        n_cur = omega_mech * 30 / np.pi
        el = wt - _wall_start[0]
        i0 = zero_seq_current(i6)
        if_str = f"  i_f={i_f:.1f} А" if itsc_active(t) else ""
        print(f"    t={t*1000:8.2f} мс  n={n_cur:7.1f} об/мин  "
              f"Lm={Lm_c*1e3:.3f} мГн  Im={Im:.1f} А  "
              f"i0={i0:.2f} А{if_str}  "
              f"calls={_call_count[0]:>9d}  wall={el:.0f}с", flush=True)
        _last_wall[0] = wt

    if ITSC_ENABLED:
        return np.concatenate([dpsi_dt, [domega_dt, dtheta_dt, dpsi_f_dt]])
    else:
        return np.concatenate([dpsi_dt, [domega_dt, dtheta_dt]])


#  Гармонический анализ

def harmonic_analysis(t_sig, x_sig, f_fund, n_periods, n_harmonics=30):
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


#  Функция запуска моделирования

def run_simulation(saturation_on, label, locked_rotor=False, t_end_override=None,
                   Ts_prof=None, Tr_prof=None,
                   itsc_enabled=False, itsc_mu=0.0, itsc_phase=0,
                   itsc_r_contact=0.001, itsc_t_start=0.0, itsc_llf=0.1e-3):
    """Запуск интегрирования и постобработка."""
    global SATURATION_ENABLED, LOCKED_ROTOR
    global ITSC_ENABLED, ITSC_MU, ITSC_PHASE, ITSC_R_contact, ITSC_T_START, ITSC_Llf

    SATURATION_ENABLED = saturation_on
    LOCKED_ROTOR = locked_rotor
    ITSC_ENABLED = itsc_enabled
    ITSC_MU = itsc_mu
    ITSC_PHASE = itsc_phase
    ITSC_R_contact = itsc_r_contact
    ITSC_T_START = itsc_t_start
    ITSC_Llf = itsc_llf

    sim_t_end = t_end_override if t_end_override is not None else t_end

    Ts_p = Ts_prof if Ts_prof is not None else Ts_profile
    Tr_p = Tr_prof if Tr_prof is not None else Tr_profile
    _Ts_func[0] = _build_temp_interp(Ts_p, sim_t_end)
    _Tr_func[0] = _build_temp_interp(Tr_p, sim_t_end)

    Ts_start = Ts_p[0]; Ts_end = Ts_p[-1]
    Tr_start = Tr_p[0]; Tr_end = Tr_p[-1]

    mode_str = "С НАСЫЩЕНИЕМ" if saturation_on else "БЕЗ НАСЫЩЕНИЯ (Lm=const)"
    lr_str = "  [ЗАТОРМОЖЕННЫЙ РОТОР]" if locked_rotor else ""
    itsc_str = ""
    if itsc_enabled:
        ph_name = ['A', 'B', 'C'][itsc_phase]
        itsc_str = (f"\n  МВКЗ: фаза {ph_name}, mu={itsc_mu*100:.1f}% витков, "
                    f"R_contact={itsc_r_contact:.4f} Ом, Llf={itsc_llf*1e3:.3f} мГн, "
                    f"t_start={itsc_t_start:.3f} с")

    _header(f"АД ADM174 -- {mode_str}{lr_str}\n"
            f"  Схема соединения: {CONN_MODE}\n"
            f"  Температура: Ts = {Ts_p} °C,  Tr = {Tr_p} °C{itsc_str}")

    if saturation_on:
        print(f"  Kнас = {K_sat}")
        print(f"  Lm_dq_nom = {Lm_dq_nom*1e3:.3f} мГн  (при Im_nom = {Im_nom_dq} А)")
        print(f"  Lm_dq(0)  = {Lm_dq_0*1e3:.3f} мГн  (ненасыщенное)")
    else:
        print(f"  Lm_3ph = {Lm_nom*1e3:.3f} мГн = const")

    print(f"\n  Интегрирование (t_end = {sim_t_end} с)...", flush=True)
    t0 = timer.time()

    if itsc_enabled:
        y0 = np.zeros(9)  # 6 psi + omega + theta + psi_f
    else:
        y0 = np.zeros(8)

    _last_Lm[0] = Lm_0 if saturation_on else Lm_nom
    _call_count[0] = 0
    _wall_start[0] = t0
    _last_wall[0] = t0

    sol = solve_ivp(
        ode_system, [0, sim_t_end], y0,
        method='LSODA', max_step=5e-5,
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
    i_f_ = np.zeros(Nt)  # Ток КЗ-контура
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
        t_k = t_arr[k]

        if itsc_enabled and itsc_active(t_k):
            psi7 = np.zeros(7)
            psi7[:6] = sol.y[:6, k]
            psi7[6] = sol.y[8, k]

            if CONN_MODE == 'star_isolated':
                i6, i_f_k, Lm_c = flux_to_currents_sat_isolated_itsc(
                    psi7, theta_e[k], Lm_c, itsc_mu, itsc_phase, itsc_llf)
            else:
                i6, i_f_k, Lm_c = flux_to_currents_sat_itsc(
                    psi7, theta_e[k], Lm_c, itsc_mu, itsc_phase, itsc_llf)
            i_f_[k] = i_f_k
        else:
            if CONN_MODE == 'star_isolated':
                i6, Lm_c = flux_to_currents_sat_isolated(sol.y[:6, k], theta_e[k], Lm_c)
            else:
                i6, Lm_c = flux_to_currents_sat(sol.y[:6, k], theta_e[k], Lm_c)
            i_f_k = 0.0

        iA_[k]=i6[0]; iB_[k]=i6[1]; iC_[k]=i6[2]
        ia_[k]=i6[3]; ib_[k]=i6[4]; ic_[k]=i6[5]

        cos_sr = np.cos(theta_e[k] + _SR_ANGLES)
        if itsc_enabled and itsc_active(t_k):
            Im_a[k] = _compute_Im_dq_itsc(i6, i_f_k, cos_sr, itsc_mu, itsc_phase)
            Me_[k] = electromagnetic_torque_itsc(i6, i_f_k, theta_e[k], Lm_c, itsc_mu, itsc_phase)
        else:
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

    # Вывод результатов
    idx_pre = np.searchsorted(t_arr, min(1.49, t_arr[-1] - 0.01))
    _subheader(f"РЕЗУЛЬТАТЫ ({label}, {CONN_MODE})")

    if locked_rotor:
        print(f"\n  > ЗАТОРМОЖЕННЫЙ РОТОР (уст.):"
              f"\n      Me = {Me_[-1]:.2f} Н*м"
              f"\n      Is = {Is_rms_[-1]:.1f} А"
              f"\n      i_n = {i_neut[-1]:.3f} А")
        if itsc_enabled:
            print(f"      i_f (КЗ-контур) = {i_f_[-1]:.1f} А")
    else:
        print(f"\n  > ХХ (t ~= 1.49 с):"
              f"\n      n = {n_rpm_arr[idx_pre]:.1f} об/мин"
              f"\n      Is = {Is_rms_[idx_pre]:.1f} А")
        if itsc_enabled:
            print(f"      i_f (КЗ-контур) = {i_f_[idx_pre]:.1f} А")
        print(f"\n  > Нагрузка (конец):"
              f"\n      n = {n_rpm_arr[-1]:.1f} об/мин"
              f"\n      Is = {Is_rms_[-1]:.1f} А")
        if itsc_enabled:
            print(f"      i_f (КЗ-контур) = {i_f_[-1]:.1f} А")

    return dict(
        t=t_arr, sol=sol,
        PsiA=sol.y[0], PsiB=sol.y[1], PsiC=sol.y[2],
        Psi_a=sol.y[3], Psi_b=sol.y[4], Psi_c=sol.y[5],
        omega=omega_m, theta=theta_e,
        n_rpm=n_rpm_arr, slip=slip_arr,
        iA=iA_, iB=iB_, iC=iC_,
        ia=ia_, ib=ib_, ic=ic_,
        i_f=i_f_,  # Ток КЗ-контура
        Me=Me_, Lm=Lm_a, Im=Im_a,
        is0=is0_, i_neutral=i_neut,
        us0=us0_, u_neutral=u_neut,
        Is_rms=Is_rms_, Ps=Ps_, Pmech=Pmech_,
        Rs=Rs_arr, Rr=Rr_arr,
        label=label,
    )


#  ЗАПУСК МОДЕЛИРОВАНИЯ

if __name__ == '__main__':

    print("\n  Контрольные точки кривой намагничивания (power-invariant dq):")
    for Im in [0, 30, 57.45, 100, 200, 500, 1000, 2000]:
        Ld = Lm_dq_of_Im(Im)
        psi = psi_m_dq(Im)
        r = Ld / Lm_dq_nom * 100
        print(f"    Im={Im:7.1f} А:  Psi_m={psi:.4f} Вб  "
              f"Lm_dq={Ld*1e3:.3f} мГн ({r:.1f}% ном)")

    #  1. ЗДОРОВЫЙ двигатель (референсный прогон)
    res_healthy = run_simulation(
        saturation_on=True, label="Здоровый (с насыщением)",
        itsc_enabled=False)

    #  2. МВКЗ -- разные степени повреждения
    mu_values = [0.02, 0.05, 0.10]  # 2%, 5%, 10% витков в КЗ
    results_itsc = {}

    for mu_val in mu_values:
        label = f"МВКЗ mu={mu_val*100:.0f}%"
        res = run_simulation(
            saturation_on=True, label=label,
            itsc_enabled=True, itsc_mu=mu_val, itsc_phase=0,  # фаза A
            itsc_r_contact=0.001, itsc_t_start=0.0)
        results_itsc[mu_val] = res

    #  ВИЗУАЛИЗАЦИЯ

    plt.rcParams.update({
        'font.size': 9, 'figure.dpi': 150, 'lines.linewidth': 0.5,
        'axes.grid': True, 'grid.alpha': 0.3,
    })
    cs = ['#d62728', '#1f77b4', '#2ca02c']

    # Рис. 1: Сравнение токов: здоровый vs МВКЗ

    fig1, axes1 = plt.subplots(len(mu_values) + 1, 3, figsize=(18, 4*(len(mu_values)+1)))
    fig1.suptitle('Сравнение токов статора: здоровый двигатель vs МВКЗ (фаза A)',
                  fontsize=13, fontweight='bold')

    # Установившийся режим ХХ -- последние 4 периода до t=1.5с
    T_period = 1.0 / f

    # Здоровый двигатель
    t_h = res_healthy['t']
    mask_xx = (t_h >= 1.42) & (t_h < 1.5)

    for col, (ph_key, ph_name, c) in enumerate([('iA','$i_A$',cs[0]),
                                                   ('iB','$i_B$',cs[1]),
                                                   ('iC','$i_C$',cs[2])]):
        ax = axes1[0, col]
        ax.plot(t_h[mask_xx]*1e3, res_healthy[ph_key][mask_xx], c=c, lw=1.0)
        ax.set_title(f'Здоровый -- {ph_name} (ХХ)')
        ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
        ax.grid(alpha=0.3)

    for row_idx, mu_val in enumerate(mu_values):
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r = (t_r >= 1.42) & (t_r < 1.5)

        for col, (ph_key, ph_name, c) in enumerate([('iA','$i_A$',cs[0]),
                                                       ('iB','$i_B$',cs[1]),
                                                       ('iC','$i_C$',cs[2])]):
            ax = axes1[row_idx + 1, col]
            ax.plot(t_r[mask_r]*1e3, res[ph_key][mask_r], c=c, lw=1.0)
            ax.set_title(f'МВКЗ {mu_val*100:.0f}% -- {ph_name} (ХХ)')
            ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('itsc_currents_comparison_xx.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: токи ХХ")

    # Рис. 2: Ток КЗ-контура

    fig2, axes2 = plt.subplots(1, len(mu_values), figsize=(6*len(mu_values), 4))
    fig2.suptitle('Ток КЗ-контура (i_f) при различных степенях МВКЗ',
                  fontsize=12, fontweight='bold')

    for idx, mu_val in enumerate(mu_values):
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r = (t_r >= 1.42) & (t_r < 1.5)
        ax = axes2[idx] if len(mu_values) > 1 else axes2
        ax.plot(t_r[mask_r]*1e3, res['i_f'][mask_r], c='#d62728', lw=1.0)
        ax.set_title(f'$i_f$ (mu={mu_val*100:.0f}%)')
        ax.set_xlabel('t, мс'); ax.set_ylabel('$i_f$, А')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('itsc_fault_current.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: ток КЗ-контура")

    # Рис. 3: Гармонический анализ -- здоровый vs МВКЗ

    _header("ГАРМОНИЧЕСКИЙ АНАЛИЗ: ЗДОРОВЫЙ vs МВКЗ")

    # ХХ-режим
    t_h = res_healthy['t']
    t_xx_end = 1.5
    mask_h_xx = t_h < t_xx_end

    _, _, harm_h_A, thd_h_A = harmonic_analysis(t_h[mask_h_xx], res_healthy['iA'][mask_h_xx], f, 4)
    _, _, harm_h_B, thd_h_B = harmonic_analysis(t_h[mask_h_xx], res_healthy['iB'][mask_h_xx], f, 4)
    _, _, harm_h_C, thd_h_C = harmonic_analysis(t_h[mask_h_xx], res_healthy['iC'][mask_h_xx], f, 4)

    _section("Здоровый двигатель -- ХХ")
    print(f"  THD:  iA={thd_h_A:.2f}%  iB={thd_h_B:.2f}%  iC={thd_h_C:.2f}%")

    n_mu = len(mu_values)
    fig3 = plt.figure(figsize=(18, 5*(n_mu + 1)))
    fig3.suptitle('Гармонический анализ: здоровый vs МВКЗ (ХХ, фаза A)',
                  fontsize=13, fontweight='bold', y=0.99)
    gs3 = GridSpec(n_mu + 1, 3, hspace=0.45, wspace=0.30,
                   left=0.06, right=0.97, top=0.94, bottom=0.04)

    # Здоровый -- спектры 3 фаз
    for col, (harm, thd, ph_name, c) in enumerate([
        (harm_h_A, thd_h_A, '$i_A$', cs[0]),
        (harm_h_B, thd_h_B, '$i_B$', cs[1]),
        (harm_h_C, thd_h_C, '$i_C$', cs[2])]):
        ax = fig3.add_subplot(gs3[0, col])
        k_arr = harm[1:16, 0].astype(int)
        a_arr = harm[1:16, 2]
        ax.bar(k_arr, a_arr, color=c, width=0.7, alpha=0.8)
        A1 = harm[1, 2]
        for i, (k, a) in enumerate(zip(k_arr, a_arr)):
            if a > 0.01 * A1:
                ax.text(k, a + A1*0.02, f'{a:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)
        ax.set_xlabel('Гармоника'); ax.set_ylabel('А')
        ax.set_title(f'Здоровый {ph_name} (THD={thd:.2f}%)')
        ax.grid(alpha=0.3, axis='y')

    # МВКЗ -- спектры
    for row_idx, mu_val in enumerate(mu_values):
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r_xx = t_r < t_xx_end

        harms = []
        thds = []
        for ph_key in ['iA', 'iB', 'iC']:
            _, _, h, thd = harmonic_analysis(t_r[mask_r_xx], res[ph_key][mask_r_xx], f, 4)
            harms.append(h)
            thds.append(thd)

        _section(f"МВКЗ mu={mu_val*100:.0f}% -- ХХ")
        print(f"  THD:  iA={thds[0]:.2f}%  iB={thds[1]:.2f}%  iC={thds[2]:.2f}%")

        # Таблица гармоник
        print(f"\n  {'Гарм':>5}  {'f,Гц':>6}  {'iA здор':>8}  {'iA МВКЗ':>8}  "
              f"{'iB здор':>8}  {'iB МВКЗ':>8}  {'iC здор':>8}  {'iC МВКЗ':>8}")
        for k_idx in range(min(16, len(harm_h_A))):
            print(f"  {k_idx:5d}  {harm_h_A[k_idx,1]:6.0f}  "
                  f"{harm_h_A[k_idx,2]:8.2f}  {harms[0][k_idx,2]:8.2f}  "
                  f"{harm_h_B[k_idx,2]:8.2f}  {harms[1][k_idx,2]:8.2f}  "
                  f"{harm_h_C[k_idx,2]:8.2f}  {harms[2][k_idx,2]:8.2f}")

        for col, (harm, thd, ph_name, c) in enumerate([
            (harms[0], thds[0], '$i_A$', cs[0]),
            (harms[1], thds[1], '$i_B$', cs[1]),
            (harms[2], thds[2], '$i_C$', cs[2])]):
            ax = fig3.add_subplot(gs3[row_idx + 1, col])
            k_arr = harm[1:16, 0].astype(int)
            a_arr = harm[1:16, 2]
            ax.bar(k_arr, a_arr, color=c, width=0.7, alpha=0.8)
            A1 = harm[1, 2]
            for i, (k, a) in enumerate(zip(k_arr, a_arr)):
                if a > 0.01 * A1:
                    ax.text(k, a + A1*0.02, f'{a:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)
            ax.set_xlabel('Гармоника'); ax.set_ylabel('А')
            ax.set_title(f'МВКЗ {mu_val*100:.0f}% {ph_name} (THD={thd:.2f}%)')
            ax.grid(alpha=0.3, axis='y')

    plt.savefig('itsc_harmonics_comparison.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: гармоники")

    # Рис. 4: Наложение iA здоровый vs МВКЗ

    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))
    fig4.suptitle('Наложение токов фазы A: здоровый vs МВКЗ',
                  fontsize=13, fontweight='bold')

    # ХХ
    t_h = res_healthy['t']
    mask_h_xx = (t_h >= 1.42) & (t_h < 1.5)

    ax = axes4[0, 0]
    ax.plot(t_h[mask_h_xx]*1e3, res_healthy['iA'][mask_h_xx],
            c='black', lw=1.5, label='Здоровый')
    for mu_val in mu_values:
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r = (t_r >= 1.42) & (t_r < 1.5)
        ax.plot(t_r[mask_r]*1e3, res['iA'][mask_r],
                lw=0.8, alpha=0.8, label=f'МВКЗ {mu_val*100:.0f}%')
    ax.set_title('$i_A$ -- ХХ'); ax.legend(fontsize=7)
    ax.set_xlabel('t, мс'); ax.set_ylabel('i, А'); ax.grid(alpha=0.3)

    # ХХ -- разность
    ax = axes4[0, 1]
    for mu_val in mu_values:
        res = results_itsc[mu_val]
        t_r = res['t']
        # Интерполируем на общую сетку
        mask_h = (t_h >= 1.42) & (t_h < 1.5)
        mask_r = (t_r >= 1.42) & (t_r < 1.5)
        t_common = np.linspace(1.42, 1.5, 2000)
        iA_h_interp = np.interp(t_common, t_h[mask_h], res_healthy['iA'][mask_h])
        iA_f_interp = np.interp(t_common, t_r[mask_r], res['iA'][mask_r])
        diff = iA_f_interp - iA_h_interp
        ax.plot(t_common*1e3, diff, lw=0.8, label=f'mu={mu_val*100:.0f}%')
    ax.set_title('$\\Delta i_A = i_A^{МВКЗ} - i_A^{здор}$ (ХХ)')
    ax.legend(fontsize=7)
    ax.set_xlabel('t, мс'); ax.set_ylabel('$\\Delta i$, А'); ax.grid(alpha=0.3)

    # Нагрузка
    mask_h_ld = (t_h >= t_h[-1] - 4*T_period)

    ax = axes4[1, 0]
    ax.plot(t_h[mask_h_ld]*1e3, res_healthy['iA'][mask_h_ld],
            c='black', lw=1.5, label='Здоровый')
    for mu_val in mu_values:
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r = (t_r >= t_r[-1] - 4*T_period)
        ax.plot(t_r[mask_r]*1e3, res['iA'][mask_r],
                lw=0.8, alpha=0.8, label=f'МВКЗ {mu_val*100:.0f}%')
    ax.set_title('$i_A$ -- Нагрузка'); ax.legend(fontsize=7)
    ax.set_xlabel('t, мс'); ax.set_ylabel('i, А'); ax.grid(alpha=0.3)

    # Нагрузка -- разность
    ax = axes4[1, 1]
    for mu_val in mu_values:
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_h = (t_h >= t_h[-1] - 4*T_period)
        mask_r = (t_r >= t_r[-1] - 4*T_period)
        t_start = max(t_h[mask_h][0], t_r[mask_r][0])
        t_stop = min(t_h[mask_h][-1], t_r[mask_r][-1])
        t_common = np.linspace(t_start, t_stop, 2000)
        iA_h_interp = np.interp(t_common, t_h[mask_h], res_healthy['iA'][mask_h])
        iA_f_interp = np.interp(t_common, t_r[mask_r], res['iA'][mask_r])
        diff = iA_f_interp - iA_h_interp
        ax.plot(t_common*1e3, diff, lw=0.8, label=f'mu={mu_val*100:.0f}%')
    ax.set_title('$\\Delta i_A$ -- Нагрузка')
    ax.legend(fontsize=7)
    ax.set_xlabel('t, мс'); ax.set_ylabel('$\\Delta i$, А'); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('itsc_overlay_iA.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: наложение iA")

    # Рис. 5: Полная картина -- пуск и нагрузка для МВКЗ 5%

    mu_demo = 0.05
    res_demo = results_itsc[mu_demo]
    t_d = res_demo['t']
    t_ms_d = t_d * 1000

    fig5 = plt.figure(figsize=(17, 22))
    fig5.suptitle(f'АД ADM174 с МВКЗ (mu={mu_demo*100:.0f}%, фаза A, {CONN_MODE})',
                  fontsize=13, fontweight='bold', y=0.995)
    gs5 = GridSpec(5, 2, hspace=0.45, wspace=0.25,
                   left=0.06, right=0.97, top=0.96, bottom=0.03)

    ax = fig5.add_subplot(gs5[0, 0])
    ax.plot(t_ms_d, res_demo['iA'], c=cs[0], lw=0.4, label='$i_A$')
    ax.plot(t_ms_d, res_demo['iB'], c=cs[1], lw=0.4, label='$i_B$')
    ax.plot(t_ms_d, res_demo['iC'], c=cs[2], lw=0.4, label='$i_C$')
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('i, А')
    ax.set_title('Токи статора (МВКЗ)'); ax.legend(fontsize=7)

    ax = fig5.add_subplot(gs5[0, 1])
    ax.plot(t_ms_d, res_demo['i_f'], c='#d62728', lw=0.5)
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('$i_f$, А')
    ax.set_title('Ток КЗ-контура')

    ax = fig5.add_subplot(gs5[1, 0])
    ax.plot(t_ms_d, res_demo['Me'], c='#9467bd', lw=0.35)
    ax.axhline(0, c='k', lw=0.3); ax.axhline(1096, c='red', ls='--', lw=0.5)
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('$M_e$, Н*м'); ax.set_title('Момент')

    ax = fig5.add_subplot(gs5[1, 1])
    ax.plot(t_ms_d, res_demo['n_rpm'], c='#ff7f0e', lw=0.7)
    ax.axhline(n_sync, c='gray', ls=':', lw=0.8)
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('n, об/мин'); ax.set_title('Скорость')

    ax = fig5.add_subplot(gs5[2, 0])
    ax.plot(t_ms_d, res_demo['Lm']*1e3, c='#e377c2', lw=0.5)
    ax.axhline(Lm_nom*1e3, c='gray', ls=':', lw=0.8)
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('$L_m$, мГн'); ax.set_title('Взаимная индуктивность')

    ax = fig5.add_subplot(gs5[2, 1])
    ax.plot(t_ms_d, res_demo['Im'], c='#17becf', lw=0.5)
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('$I_m$ (dq), А'); ax.set_title('Ток намагничивания')

    ax = fig5.add_subplot(gs5[3, 0])
    ax.plot(t_ms_d, res_demo['Is_rms'], c='#17becf', lw=0.4)
    ax.axhline(57.45, c='green', ls=':', lw=0.8, label='$I_0$')
    ax.axhline(204.42, c='gray', ls=':', lw=0.8, label='$I_n$')
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('$I_s$, А'); ax.set_title('Действ. ток')
    ax.legend(fontsize=7)

    ax = fig5.add_subplot(gs5[3, 1])
    ax.plot(t_ms_d, res_demo['i_neutral'], c='#ff6600', lw=0.5)
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('$i_n$, А')
    ax.set_title(f'Ток нейтрали ({CONN_MODE})')

    ax = fig5.add_subplot(gs5[4, 0])
    ax.plot(t_ms_d, res_demo['Ps']/1e3, c='#e377c2', lw=0.3, label='$P_{эл}$')
    ax.plot(t_ms_d, res_demo['Pmech']/1e3, c='#bcbd22', lw=0.3, label='$P_{мех}$')
    ax.axhline(170, c='gray', ls=':', lw=0.8)
    ax.axvline(1500, c='red', lw=0.5, ls=':')
    ax.set_xlabel('t, мс'); ax.set_ylabel('P, кВт'); ax.set_title('Мощности')
    ax.legend(fontsize=7)

    ax = fig5.add_subplot(gs5[4, 1])
    ax.plot(res_demo['n_rpm'], res_demo['Me'], c='#9467bd', lw=0.15, alpha=0.6)
    ax.axhline(0, c='k', lw=0.3)
    ax.set_xlabel('n, об/мин'); ax.set_ylabel('$M_e$, Н*м')
    ax.set_title('Мех. характеристика')

    plt.savefig('itsc_full_picture.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: полная картина")

    _header("ГОДОГРАФ ТОКА СТАТОРА")

    T_period = 1.0 / f
    _SQRT2_3 = np.sqrt(2.0/3.0)

    def to_alpha_beta(iA, iB, iC):
        """Преобразование Кларк (power-invariant)."""
        i_alpha = _SQRT2_3 * (iA - 0.5*iB - 0.5*iC)
        i_beta  = _SQRT2_3 * (np.sqrt(3)/2.0) * (iB - iC)
        return i_alpha, i_beta
    fig6a, axes6a = plt.subplots(1, 1 + len(mu_values),
                                  figsize=(5*(1+len(mu_values)), 5))
    fig6a.suptitle('Годограф тока статора (a-b) - ХХ, установившийся режим',
                   fontsize=13, fontweight='bold')

    # Здоровый
    t_h = res_healthy['t']
    mask_h = (t_h >= 1.42) & (t_h < 1.5)
    ia_h, ib_h = to_alpha_beta(res_healthy['iA'][mask_h],
                                res_healthy['iB'][mask_h],
                                res_healthy['iC'][mask_h])

    ax = axes6a[0]
    ax.plot(ia_h, ib_h, c='black', lw=0.8)
    ax.set_xlabel(r'$i_a$, А'); ax.set_ylabel(r'$i_b$, А')
    ax.set_title('Здоровый')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.axhline(0, c='k', lw=0.3); ax.axvline(0, c='k', lw=0.3)

    for idx, mu_val in enumerate(mu_values):
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r = (t_r >= 1.42) & (t_r < 1.5)
        ia_f, ib_f = to_alpha_beta(res['iA'][mask_r],
                                    res['iB'][mask_r],
                                    res['iC'][mask_r])

        ax = axes6a[idx + 1]
        # Здоровый -- тонкой линией для сравнения
        ax.plot(ia_h, ib_h, c='gray', lw=0.5, ls='--', alpha=0.5, label='здоров.')
        # Неисправный
        ax.plot(ia_f, ib_f, c=cs[0], lw=0.8, label=f'μ={mu_val*100:.0f}%')
        ax.set_xlabel(r'$i_a$, А'); ax.set_ylabel(r'$i_b$, А')
        ax.set_title(f'МВКЗ {mu_val*100:.0f}%')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.axhline(0, c='k', lw=0.3); ax.axvline(0, c='k', lw=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('itsc_hodograph_xx.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: годограф ХХ")
    fig6b, ax6b = plt.subplots(1, 1, figsize=(8, 8))
    ax6b.set_title('Годограф тока статора (a-b) - ХХ, сравнение',
                    fontsize=12, fontweight='bold')

    ax6b.plot(ia_h, ib_h, c='black', lw=1.5, label='Здоровый')

    colors_mu = ['#2ca02c', '#ff7f0e', '#d62728']
    for idx, mu_val in enumerate(mu_values):
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r = (t_r >= 1.42) & (t_r < 1.5)
        ia_f, ib_f = to_alpha_beta(res['iA'][mask_r],
                                    res['iB'][mask_r],
                                    res['iC'][mask_r])
        ax6b.plot(ia_f, ib_f, c=colors_mu[idx], lw=0.8,
                  label=f'МВКЗ {mu_val*100:.0f}%', alpha=0.8)

    ax6b.set_xlabel(r'$i_a$, А', fontsize=11)
    ax6b.set_ylabel(r'$i_b$, А', fontsize=11)
    ax6b.set_aspect('equal')
    ax6b.grid(alpha=0.3)
    ax6b.axhline(0, c='k', lw=0.3); ax6b.axvline(0, c='k', lw=0.3)
    ax6b.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.savefig('itsc_hodograph_combined.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: годограф совмещённый")
    fig6c, axes6c = plt.subplots(1, 1 + len(mu_values),
                                  figsize=(5*(1+len(mu_values)), 5))
    fig6c.suptitle('Годограф тока статора (a-b) - Нагрузка, установившийся режим',
                   fontsize=13, fontweight='bold')

    mask_h_ld = (t_h >= t_h[-1] - 4*T_period)
    ia_h_ld, ib_h_ld = to_alpha_beta(res_healthy['iA'][mask_h_ld],
                                      res_healthy['iB'][mask_h_ld],
                                      res_healthy['iC'][mask_h_ld])

    ax = axes6c[0]
    ax.plot(ia_h_ld, ib_h_ld, c='black', lw=0.8)
    ax.set_xlabel(r'$i_a$, А'); ax.set_ylabel(r'$i_b$, А')
    ax.set_title('Здоровый')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.axhline(0, c='k', lw=0.3); ax.axvline(0, c='k', lw=0.3)

    for idx, mu_val in enumerate(mu_values):
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r = (t_r >= t_r[-1] - 4*T_period)
        ia_f, ib_f = to_alpha_beta(res['iA'][mask_r],
                                    res['iB'][mask_r],
                                    res['iC'][mask_r])

        ax = axes6c[idx + 1]
        ax.plot(ia_h_ld, ib_h_ld, c='gray', lw=0.5, ls='--', alpha=0.5, label='здоров.')
        ax.plot(ia_f, ib_f, c=cs[0], lw=0.8, label=f'μ={mu_val*100:.0f}%')
        ax.set_xlabel(r'$i_a$, А'); ax.set_ylabel(r'$i_b$, А')
        ax.set_title(f'МВКЗ {mu_val*100:.0f}%')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.axhline(0, c='k', lw=0.3); ax.axvline(0, c='k', lw=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('itsc_hodograph_load.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: годограф нагрузка")
    fig6d, axes6d = plt.subplots(1, len(mu_values),
                                  figsize=(5*len(mu_values), 5))
    fig6d.suptitle('Фазовый портрет: $i_f$ vs $i_A$ - ХХ',
                   fontsize=13, fontweight='bold')

    for idx, mu_val in enumerate(mu_values):
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r = (t_r >= 1.42) & (t_r < 1.5)

        ax = axes6d[idx] if len(mu_values) > 1 else axes6d
        ax.plot(res['iA'][mask_r], res['i_f'][mask_r], c=cs[0], lw=0.8)
        ax.set_xlabel('$i_A$, А'); ax.set_ylabel('$i_f$, А')
        ax.set_title(f'μ={mu_val*100:.0f}%')
        ax.grid(alpha=0.3)
        ax.axhline(0, c='k', lw=0.3); ax.axvline(0, c='k', lw=0.3)

    plt.tight_layout()
    plt.savefig('itsc_phase_portrait_if_iA.png', dpi=150, bbox_inches='tight')
    _fig_saved("МВКЗ: фазовый портрет i_f vs i_A")

    # Рис. 7: Сводная таблица THD

    _header("СВОДНАЯ ТАБЛИЦА THD")

    print(f"\n  {'Режим':>20}  {'iA THD,%':>10}  {'iB THD,%':>10}  {'iC THD,%':>10}")
    print(f"  {'-'*55}")

    # Здоровый
    print(f"  {'Здоровый':>20}  {thd_h_A:10.2f}  {thd_h_B:10.2f}  {thd_h_C:10.2f}")

    for mu_val in mu_values:
        res = results_itsc[mu_val]
        t_r = res['t']
        mask_r_xx = t_r < t_xx_end
        _, _, h_A, thd_A = harmonic_analysis(t_r[mask_r_xx], res['iA'][mask_r_xx], f, 4)
        _, _, h_B, thd_B = harmonic_analysis(t_r[mask_r_xx], res['iB'][mask_r_xx], f, 4)
        _, _, h_C, thd_C = harmonic_analysis(t_r[mask_r_xx], res['iC'][mask_r_xx], f, 4)
        label = f"МВКЗ {mu_val*100:.0f}%"
        print(f"  {label:>20}  {thd_A:10.2f}  {thd_B:10.2f}  {thd_C:10.2f}")

    _header("Готово.")