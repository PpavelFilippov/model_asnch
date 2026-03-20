import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List


# ПАРАМЕТРЫ

@dataclass
class SGParams:
    """Параметры одного синхронного генератора (дизель-генератор)."""
    name: str = "СГ"
    S_nom: float = 400.0  # кВА
    P_nom: float = 320.0  # кВт
    V_nom: float = 1.0  # Номинальное напряжение, о.е.
    H: float = 3.5  # Постоянная инерции, с
    D_mech: float = 2.0  # Коэффициент демпфирования, о.е.

    # Электрические параметры (о.е. на базе S_nom)
    Xd: float = 1.2  # Синхронное реактивное сопротивление d
    Xq: float = 0.8  # Синхронное реактивное сопротивление q
    Xd_prime: float = 0.3  # Переходное сопротивление d
    Xd_dprime: float = 0.18  # Субпереходное сопротивление d
    Td0_prime: float = 5.0  # Постоянная времени разомкнутой обмотки возб., с
    Td0_dprime: float = 0.035  # Постоянная субпереходная, с
    Ra: float = 0.01  # Сопротивление статора, о.е.

    # Губернатор - модель TGOV1 (IEEE)
    R_droop: float = 0.05  # Статизм (droop), о.е.
    T_act: float = 0.1  # Постоянная времени актуатора (сервомотор), с
    T_gov: float = 0.5  # Постоянная времени турбины/дизеля, с
    ramp_up: float = 0.08  # Макс. скорость набора мощности, о.е./с
    ramp_down: float = 0.15  # Макс. скорость сброса мощности, о.е./с
    P_min: float = 0.05  # Минимальная мощность, о.е.
    P_max: float = 1.0  # Максимальная мощность, о.е.

    # АРВ - упрощённая модель SEXS (IEEE Std 421.5)
    Ka: float = 100.0  # Коэффициент усиления регулятора
    Ta: float = 0.01  # Постоянная времени регулятора, с
    Tb: float = 10.0  # Числитель lead-lag, с
    Tc: float = 1.0  # Знаменатель lead-lag, с
    Ke: float = 1.0  # Коэффициент возбудителя
    Te: float = 0.5  # Постоянная времени возбудителя, с
    Vref: float = 1.0  # Уставка напряжения
    Efd_min: float = 0.0  # Мин. напряжение возбуждения
    Efd_max: float = 5.0  # Макс. напряжение возбуждения
    Xc: float = 0.05  # Компенсация по реактивному току

    X_line_sg: float = 0.0  # Реактивное, о.е. на базе S_nom (0 = прямое подключение)
    R_line_sg: float = 0.0  # Активное, о.е. на базе S_nom


@dataclass
class VSGParams:
    """Параметры ВСГ (инвертор + BESS)."""
    name: str = "ВСГ"
    S_nom: float = 300.0  # кВА
    P_nom: float = 250.0  # кВт

    # Виртуальная механика
    H_virt: float = 1.5  # Виртуальная постоянная инерции, с
    D_virt: float = 5.0  # Виртуальное демпфирование, о.е.

    # P-f droop
    R_droop: float = 0.03  # Статизм по частоте, о.е. (3%)

    # RoCoF (df/dt) feedforward - инерционный отклик
    K_rocof: float = 0.0  # Коэффициент усиления по df/dt, о.е.*с
    T_rocof: float = 0.02  # Фильтр df/dt, с

    # Feedforward по дефициту мощности (P_load - P_sg)
    K_ff: float = 0.0  # Коэффициент feedforward (0 = отключён, 1 = полная компенсация)

    T_inv: float = 0.005  # Характерное время отклика инвертора, с
    zeta_inv: float = 0.7  # Коэффициент демпфирования
    I_max: float = 1.2  # Максимальный ток инвертора, о.е.
    P_max: float = 1.0  # Макс. P, о.е.
    P_min: float = -0.5  # Мин. P, о.е.

    # Q-V регулятор - PI
    Kp_v: float = 10.0  # Пропорциональный
    Ki_v: float = 50.0  # Интегральный
    T_qv: float = 0.005  # Постоянная времени контура Q, с
    Vref: float = 1.0  # Уставка напряжения

    Xv: float = 0.10  # Виртуальный реактанс, о.е.

    # Батарея
    E_batt_kWh: float = 200.0
    SoC_init: float = 0.80
    SoC_min: float = 0.10
    SoC_max: float = 0.95
    eta_charge: float = 0.95
    eta_discharge: float = 0.95

    T_restore: float = 0.0  # с (0 = выключено). Типично 3–10 с.

    T_washout: float = 0.0  # с (0 = без washout, feedforward DC)


@dataclass
class BusParams:
    """Параметры шины."""
    f_nom: float = 50.0
    V_nom: float = 1.0
    X_line: float = 0.05  # Реактивное сопротивление связи, о.е.
    R_line: float = 0.01  # Активное сопротивление связи, о.е.
    D_load: float = 1.0  # Зависимость нагрузки от частоты, о.е.
    # (P_load ~ P0 * (1 + D_load * df/f0))


@dataclass
class LoadProfile:
    events: List[tuple] = field(default_factory=lambda: [
        (0.0, 400.0, 200.0),
        (5.0, 700.0, 350.0),
        (20.0, 500.0, 250.0),
        (35.0, 900.0, 450.0),
        (50.0, 600.0, 300.0),
    ])


@dataclass
class SimParams:
    dt: float = 0.001
    T_end: float = 60.0
    downsample: int = 10


# МОДЕЛЬ СГ

class SynchronousGenerator:
    """
    Синхронный генератор - модель 4-го порядка.

    Состояния:
      1) delta - угол ротора
      2) omega - угловая скорость
      3) Eq_prime - переходная ЭДС (медленная, T'd0 ≈ 5 с)
      4) Eq_dprime - субпереходная ЭДС (быстрая, T''d0 ≈ 35 мс)
      5) Efd, Vr - возбудитель и АРВ
      6) P_valve, P_mech - губернатор

    Мощность считается через E''q и X''d - даёт правильный
    бросок тока в первые 50-100 мс после наброса нагрузки.
    """

    def __init__(self, params: SGParams, idx: int = 0):
        self.p = params
        self.idx = idx
        self.w0 = 2 * np.pi * 50.0

        self.delta = 0.0
        self.omega = self.w0
        self.Eq_prime = 1.0
        self.Eq_dprime = 1.0  # Субпереходная ЭДС
        self.Efd = 1.0
        self.Efd0 = 1.0  # Рабочая точка возбуждения (вычисляется при init)
        self.Vr = 0.0
        self.P_valve = 0.0
        self.P_mech = 0.0
        self.P_set = 0.0

        self.Pe = 0.0
        self.Qe = 0.0
        self.Vt = 1.0
        self.Id = 0.0
        self.Iq = 0.0

    def init_steady_state(self, P0_pu, Q0_pu, V_bus):
        """
        Полная инициализация установившегося режима.
        Критично: step() считает Pe через E''q и X_total, поэтому
        инициализация должна обеспечить Pe(t=0) = P0 именно через эту формулу.
        """
        p = self.p
        Vt = V_bus
        self.Vt = Vt
        self.Pe = P0_pu
        self.Qe = Q0_pu

        X_total = p.Xd_dprime + p.X_line_sg

        denom = Q0_pu + Vt ** 2 / X_total
        if abs(denom) > 1e-9:
            self.delta = np.arctan2(P0_pu, denom)
        else:
            self.delta = 0.0

        sin_d = np.sin(self.delta)
        cos_d = np.cos(self.delta)

        if abs(sin_d) > 1e-9:
            self.Eq_dprime = P0_pu * X_total / (Vt * sin_d)
        else:
            self.Eq_dprime = Vt + Q0_pu * X_total / (Vt + 1e-9)

        self.Id = (self.Eq_dprime - Vt * cos_d) / X_total
        self.Iq = Vt * sin_d / X_total

        self.Eq_prime = self.Eq_dprime + (p.Xd_prime - p.Xd_dprime) * self.Id

        self.Efd = self.Eq_prime + (p.Xd - p.Xd_prime) * self.Id
        self.Efd = np.clip(self.Efd, p.Efd_min, p.Efd_max)
        self.Efd0 = self.Efd

        self.P_mech = P0_pu
        self.P_valve = P0_pu
        self.P_set = P0_pu
        self.omega = self.w0

    def step(self, dt, V_bus, P_set_external, f_bus=None):
        """Один шаг интегрирования.
        f_bus - частота шины (Гц). Если задана, губернатор измеряет её, а не частоту ротора.
        """
        p = self.p
        w0 = self.w0

        X_total = p.Xd_dprime + p.X_line_sg
        R_total = p.Ra + p.R_line_sg

        if V_bus > 0.01:
            sin_d = np.sin(self.delta)
            cos_d = np.cos(self.delta)

            self.Pe = (self.Eq_dprime * V_bus / X_total) * sin_d
            self.Qe = (self.Eq_dprime * V_bus * cos_d - V_bus ** 2) / X_total

            self.Id = (self.Eq_dprime - V_bus * cos_d) / X_total
            self.Iq = V_bus * sin_d / X_total

            Vd_t = V_bus * cos_d + R_total * self.Id - p.X_line_sg * self.Iq
            Vq_t = V_bus * sin_d + R_total * self.Iq + p.X_line_sg * self.Id
            self.Vt = np.sqrt(Vd_t ** 2 + Vq_t ** 2)
        else:
            self.Pe = self.Qe = self.Id = self.Iq = 0.0
            self.Vt = 0.0

        if f_bus is not None:
            dw_gov = (f_bus - 50.0) / 50.0  # отклонение частоты шины, о.е.
        else:
            dw_gov = (self.omega - w0) / w0  # fallback: частота ротора

        P_gov_ref = P_set_external - dw_gov / p.R_droop
        P_gov_ref = np.clip(P_gov_ref, p.P_min, p.P_max)

        dP_valve_dt = (P_gov_ref - self.P_valve) / p.T_act
        dP_valve_dt = np.clip(dP_valve_dt, -p.ramp_down, p.ramp_up)
        self.P_valve += dP_valve_dt * dt
        self.P_valve = np.clip(self.P_valve, p.P_min, p.P_max)

        dP_mech_dt = (self.P_valve - self.P_mech) / p.T_gov
        self.P_mech += dP_mech_dt * dt
        self.P_mech = np.clip(self.P_mech, p.P_min, p.P_max)

        dw_pu = (self.omega - w0) / w0
        P_accel = self.P_mech - self.Pe - p.D_mech * dw_pu
        d_omega = (P_accel / (2.0 * p.H)) * w0 * dt
        self.omega += d_omega
        self.delta += (self.omega - w0) * dt
        self.delta = (self.delta + np.pi) % (2 * np.pi) - np.pi

        V_err = p.Vref - self.Vt
        Efd_target = self.Efd0 + p.Ka * V_err
        Efd_target = np.clip(Efd_target, p.Efd_min, p.Efd_max)

        dEfd = (Efd_target - self.Efd) / p.Te * dt
        self.Efd += dEfd
        self.Efd = np.clip(self.Efd, p.Efd_min, p.Efd_max)

        dEq_p = (self.Efd - self.Eq_prime
                 - (p.Xd - p.Xd_prime) * self.Id) / p.Td0_prime
        self.Eq_prime += dEq_p * dt

        dEq_dp = (self.Eq_prime - self.Eq_dprime
                  - (p.Xd_prime - p.Xd_dprime) * self.Id) / p.Td0_dprime
        self.Eq_dprime += dEq_dp * dt

        return self.Pe * self.p.S_nom, self.Qe * self.p.S_nom


# МОДЕЛЬ ВСГ

class VirtualSynchronousGenerator:
    """
    Архитектура управления (grid-forming):
      1. Виртуальное swing equation -> виртуальная частота omega_virt
      2. P-f droop -> уставка активной мощности P_ref
      3. Инвертор отслеживает P_ref с постоянной T_inv
      4. Q-V PI-регулятор -> уставка реактивной мощности Q_ref
      5. Ограничения по S_nom и SoC батареи
    """

    def __init__(self, params: VSGParams):
        self.p = params
        self.w0 = 2 * np.pi * 50.0

        # --- Состояния ---
        self.omega_virt = self.w0
        self.delta_virt = 0.0
        self.P_ref = 0.0  # Уставка от swing equation + droop
        self.P_out = 0.0  # Фактический выход P инвертора, о.е.
        self.Q_ref = 0.0  # Уставка Q от PI-регулятора
        self.Q_out = 0.0  # Фактический выход Q инвертора, о.е.
        self.SoC = params.SoC_init
        self.V_int_err = 0.0  # Интеграл ошибки V для PI
        self.P_set = 0.0  # Внешняя уставка (диспетчер)
        self.f_prev = 50.0  # Предыдущая частота для df/dt
        self.dfdt_filt = 0.0  # Фильтрованный df/dt
        self.P_deficit_ff = 0.0  # Feedforward от дефицита мощности, о.е.
        self.dP_out = 0.0  # Скорость изменения P_out (для модели 2-го порядка)
        self.P_restore = 0.0  # Интегральный сигнал возврата к нулю
        self.P_ff_filt = 0.0  # Состояние фильтра washout для feedforward

    def init_steady_state(self, P0_pu, Q0_pu, V_bus):
        """
        Инициализация.
        V_int_err должен быть таким, чтобы PI-регулятор при V_bus = Vref
        выдавал Q_ref = Q0_pu (иначе ВСГ сбросит Q и напряжение просядет).
        Q_ref = Kp*(Vref-V) + Ki*V_int_err = 0 + Ki*V_int_err = Q0_pu
        => V_int_err = Q0_pu / Ki
        """
        self.P_out = P0_pu
        self.P_ref = P0_pu
        self.P_set = P0_pu
        self.Q_out = Q0_pu
        self.Q_ref = Q0_pu
        self.omega_virt = self.w0
        self.delta_virt = 0.0
        self.V_int_err = Q0_pu / (self.p.Ki_v + 1e-9)

    def step(self, dt, V_bus, f_bus):
        """
        Один шаг. Агрессивное управление ВСГ:
          1. Droop - пропорциональный отклик на отклонение частоты
          2. RoCoF (df/dt) - инерционный отклик на СКОРОСТЬ изменения частоты
          3. Feedforward - прямая компенсация дефицита (P_load - P_sg)
          4. Виртуальное swing equation - синхронизация
          5. Инвертор - быстрое отслеживание за 5 мс
        """
        p = self.p
        w0 = self.w0

        f_bus_hz = f_bus
        df_bus_pu = (f_bus_hz - 50.0) / 50.0
        P_droop = -df_bus_pu / p.R_droop

        if dt > 1e-9:
            dfdt_raw = (f_bus_hz - self.f_prev) / dt  # Гц/с
        else:
            dfdt_raw = 0.0
        self.f_prev = f_bus_hz

        alpha_rocof = dt / (p.T_rocof + dt)
        self.dfdt_filt += alpha_rocof * (dfdt_raw - self.dfdt_filt)

        dfdt_pu = self.dfdt_filt / 50.0
        P_rocof = -p.K_rocof * dfdt_pu

        P_deficit_raw = p.K_ff * self.P_deficit_ff

        if p.T_washout > 0.01:
            # Обновляем фильтр (low-pass часть)
            alpha_wo = dt / (p.T_washout + dt)
            self.P_ff_filt += alpha_wo * (P_deficit_raw - self.P_ff_filt)
            # Washout = вход - low-pass = высокочастотная часть
            P_ff = P_deficit_raw - self.P_ff_filt
        else:
            P_ff = P_deficit_raw

        dw_pu = (self.omega_virt - w0) / w0
        P_total_ref = self.P_set + P_droop + P_rocof + P_ff
        P_accel = P_total_ref - self.P_out - p.D_virt * dw_pu

        if p.H_virt > 0.01:
            d_omega = (P_accel / (2.0 * p.H_virt)) * w0 * dt
        else:
            d_omega = 0.0

        self.omega_virt += d_omega
        self.delta_virt += (self.omega_virt - w0) * dt

        if p.T_restore > 0.01:
            self.P_restore += (self.P_out - self.P_set) / p.T_restore * dt
            self.P_restore = np.clip(self.P_restore, -1.0, 1.0)
        else:
            self.P_restore = 0.0

        self.P_ref = P_total_ref - self.P_restore
        self.P_ref = np.clip(self.P_ref, p.P_min, p.P_max)

        if self.SoC <= p.SoC_min and self.P_ref > 0:
            self.P_ref = 0.0
        elif self.SoC >= p.SoC_max and self.P_ref < 0:
            self.P_ref = 0.0

        omega_n = 2.0 * np.pi / (4.0 * p.T_inv)
        zeta = p.zeta_inv

        ddP = omega_n ** 2 * (self.P_ref - self.P_out) - 2.0 * zeta * omega_n * self.dP_out
        self.dP_out += ddP * dt
        self.P_out += self.dP_out * dt
        self.P_out = np.clip(self.P_out, p.P_min, p.P_max)

        V_error = p.Vref - V_bus

        self.V_int_err += V_error * dt
        Q_max_avail = np.sqrt(max(0, p.I_max ** 2 - self.P_out ** 2))
        if (self.Q_out >= Q_max_avail and V_error > 0) or \
                (self.Q_out <= -Q_max_avail and V_error < 0):
            self.V_int_err -= V_error * dt  # Откатываем
        self.V_int_err = np.clip(self.V_int_err, -1.0, 1.0)

        self.Q_ref = p.Kp_v * V_error + p.Ki_v * self.V_int_err

        dQ = (self.Q_ref - self.Q_out) / p.T_qv * dt
        self.Q_out += dQ

        S_out = np.sqrt(self.P_out ** 2 + self.Q_out ** 2)
        if S_out > p.I_max:
            Q_max = np.sqrt(max(0, p.I_max ** 2 - self.P_out ** 2))
            self.Q_out = np.clip(self.Q_out, -Q_max, Q_max)

        P_batt_kW = self.P_out * p.P_nom
        if P_batt_kW >= 0:
            energy_kWh = P_batt_kW / p.eta_discharge * (dt / 3600.0)
        else:
            energy_kWh = P_batt_kW * p.eta_charge * (dt / 3600.0)

        self.SoC -= energy_kWh / p.E_batt_kWh
        self.SoC = np.clip(self.SoC, 0.0, 1.0)

        P_kW = self.P_out * p.P_nom
        Q_kvar = self.Q_out * p.S_nom
        return P_kW, Q_kvar


class BusModel:
    """
    Модель общей шины - Center of Inertia (CoI) для частоты,
    Q-V чувствительность для напряжения.

    Дополнительно моделирует связь быстрых изменений P ВСГ с
    напряжением шины через виртуальный / выходной импеданс:
      - Быстрое dP_vsg/dt -> кратковременный dV на шине
      - Это возмущает Q всех СГ через АРВ -> циркуляция Q
      - Источник: разница Xv (ВСГ) << X''d (СГ)
    Литература: Nature Sci. Rep. 2023, doi:10.1038/s41598-023-39121-6
    """

    def __init__(self, params: BusParams):
        self.p = params
        self.f = params.f_nom
        self.V = params.V_nom
        self.w0 = 2 * np.pi * params.f_nom
        self.P_vsg_prev = 0.0  # Для вычисления dP_vsg/dt

    def step(self, dt, P_gen_total_kW, Q_gen_total_kvar,
             P_load_base_kW, Q_load_base_kvar,
             H_total, S_total_kVA,
             omega_coi,
             P_vsg_kW=0.0, Q_vsg_kvar=0.0, Xv_vsg=0.1):
        """
        Частота и напряжение шины.

        Новый аргумент Xv_vsg - виртуальный реактанс ВСГ.
        Быстрое изменение P_vsg создаёт dV пропорциональное dP/dt * Xv,
        что вызывает электромагнитный переходный на шине.
        """
        p = self.p
        f_nom = p.f_nom
        w0 = self.w0

        df_pu = (self.f - f_nom) / f_nom
        P_load_kW = P_load_base_kW * (1.0 + p.D_load * df_pu)
        Q_load_kvar = Q_load_base_kvar

        P_imbalance_pu = (P_gen_total_kW - P_load_kW) / (S_total_kVA + 1e-9)
        D_sys = 2.0
        df_dt = (P_imbalance_pu - D_sys * df_pu) / (2.0 * max(H_total, 0.5)) * f_nom
        self.f += df_dt * dt
        self.f = np.clip(self.f, 47.0, 53.0)

        dP_pu = (P_gen_total_kW - P_load_kW) / (S_total_kVA + 1e-9)
        Q_imbalance_pu = (Q_gen_total_kvar - Q_load_kvar) / (S_total_kVA + 1e-9)

        if dt > 1e-9:
            dPdt_vsg = (P_vsg_kW - self.P_vsg_prev) / dt  # кВт/с
        else:
            dPdt_vsg = 0.0
        self.P_vsg_prev = P_vsg_kW
        dPdt_vsg_pu = dPdt_vsg / (S_total_kVA + 1e-9)  # о.е./с
        T_emf = 0.02
        V_emf_transient = Xv_vsg * dPdt_vsg_pu * T_emf

        V_target = (p.V_nom
                    + p.R_line * dP_pu
                    + p.X_line * Q_imbalance_pu
                    + V_emf_transient)

        T_v = 0.02
        self.V += (V_target - self.V) / T_v * dt
        self.V = np.clip(self.V, 0.85, 1.15)

        return self.f, self.V, P_load_kW, Q_load_kvar


# ОСНОВНАЯ СИМУЛЯЦИЯ

def get_load_at_time(t, load_profile: LoadProfile):
    """Ступенчатый профиль нагрузки."""
    P, Q = load_profile.events[0][1], load_profile.events[0][2]
    for ev_t, ev_P, ev_Q in load_profile.events:
        if t >= ev_t:
            P, Q = ev_P, ev_Q
        else:
            break
    return P, Q


def run_simulation(
        n_sg: int = 3,
        sg_params=None,
        vsg_params: VSGParams = None,
        bus_params: BusParams = None,
        load_profile: LoadProfile = None,
        sim_params: SimParams = None,
        P0_sg_pu: list = None,
        P_set_schedule: list = None,

):
    if sg_params is None: sg_params = SGParams()
    if vsg_params is None: vsg_params = VSGParams()
    if bus_params is None: bus_params = BusParams()
    if load_profile is None: load_profile = LoadProfile()
    if sim_params is None: sim_params = SimParams()

    if isinstance(sg_params, list):
        sg_params_list = sg_params
        assert len(sg_params_list) == n_sg, \
            f"Длина списка sg_params ({len(sg_params_list)}) != n_sg ({n_sg})"
    else:
        sg_params_list = [sg_params] * n_sg

    dt = sim_params.dt
    N = int(sim_params.T_end / dt)
    ds = sim_params.downsample
    w0 = 2 * np.pi * bus_params.f_nom

    sg_list = [SynchronousGenerator(sg_params_list[i], idx=i) for i in range(n_sg)]
    vsg = VirtualSynchronousGenerator(vsg_params)
    bus = BusModel(bus_params)

    P0_load, Q0_load = get_load_at_time(0, load_profile)
    S_sg_total = sum(sp.S_nom for sp in sg_params_list)
    S_total = S_sg_total + vsg_params.S_nom

    if P0_sg_pu is not None:
        assert len(P0_sg_pu) == n_sg
        for i, sg in enumerate(sg_list):
            sp = sg_params_list[i]
            sg_P0 = P0_sg_pu[i]  # уже в о.е. на базе S_nom
            sg_Q0 = Q0_load * (sp.S_nom / S_total) / sp.S_nom
            sg.init_steady_state(sg_P0, sg_Q0, bus_params.V_nom)
    else:
        for i, sg in enumerate(sg_list):
            sp = sg_params_list[i]
            sg_P0 = (P0_load * sp.S_nom / S_sg_total) / sp.S_nom
            sg_Q0 = Q0_load * (sp.S_nom / S_total) / sp.S_nom
            sg.init_steady_state(sg_P0, sg_Q0, bus_params.V_nom)

    vsg_P0 = 0.0
    vsg_Q0 = Q0_load * (vsg_params.S_nom / S_total) / vsg_params.S_nom
    vsg.init_steady_state(vsg_P0, vsg_Q0, bus_params.V_nom)

    H_total = (sum(sp.H * sp.S_nom for sp in sg_params_list) +
               vsg_params.H_virt * vsg_params.S_nom) / S_total

    P_set_fixed = []
    for i in range(n_sg):
        if P0_sg_pu is not None:
            P_set_fixed.append(P0_sg_pu[i])
        else:
            sp = sg_params_list[i]
            P_set_fixed.append((P0_load * sp.S_nom / S_sg_total) / sp.S_nom)

    n_out = N // ds + 2
    out = {
        't': np.zeros(n_out),
        'f': np.zeros(n_out),
        'V': np.zeros(n_out),
        'P_sg_total': np.zeros(n_out),
        'Q_sg_total': np.zeros(n_out),
        'P_sg_each': np.zeros((n_sg, n_out)),
        'Q_sg_each': np.zeros((n_sg, n_out)),
        'P_mech': np.zeros((n_sg, n_out)),
        'Efd': np.zeros((n_sg, n_out)),
        'Vt_sg': np.zeros((n_sg, n_out)),
        'P_vsg': np.zeros(n_out),
        'Q_vsg': np.zeros(n_out),
        'P_load': np.zeros(n_out),
        'Q_load': np.zeros(n_out),
        'SoC': np.zeros(n_out),
        'omega_vsg': np.zeros(n_out),
    }
    oi = 0  # output index

    # ЦИКЛ ИНТЕГРИРОВАНИЯ
    for k in range(N):
        t = k * dt
        V_bus = bus.V
        f_bus = bus.f

        P_load_base, Q_load_base = get_load_at_time(t, load_profile)

        P_sg_sum_kW = 0.0
        Q_sg_sum_kvar = 0.0

        event_happened = False
        if P_set_schedule is not None:
            for t_ev, _ in P_set_schedule:
                if t >= t_ev:
                    event_happened = True
                    break

        for i, sg in enumerate(sg_list):
            sp = sg_params_list[i]
            P_set = P_set_fixed[i]
            if P_set_schedule is not None:
                for t_ev, p_set_list in P_set_schedule:
                    if t >= t_ev:
                        P_set = p_set_list[i]
            if P_set < 0.001:
                sg.Pe = 0.0
                sg.Qe = 0.0
                sg.P_mech = 0.0
                sg.P_valve = 0.0
                sg.omega = sg.w0
                Pe_kW = 0.0
                Qe_kvar = 0.0
            elif not event_happened:
                P_set = np.clip(P_set, sp.P_min, sp.P_max)
                sg.Pe = P_set
                sg.P_mech = P_set
                sg.P_valve = P_set
                Pe_kW = P_set * sp.S_nom
                Qe_kvar = sg.Qe * sp.S_nom
            else:
                P_set = np.clip(P_set, sp.P_min, sp.P_max)
                Pe_kW, Qe_kvar = sg.step(dt, V_bus, P_set, f_bus=f_bus)

            P_sg_sum_kW += Pe_kW
            Q_sg_sum_kvar += Qe_kvar


        vsg.P_set = 0.0
        P_deficit_kW = P_load_base - P_sg_sum_kW
        vsg.P_deficit_ff = max(0.0, P_deficit_kW) / vsg_params.P_nom

        P_vsg_kW, Q_vsg_kvar = vsg.step(dt, V_bus, f_bus)


        P_gen_total = P_sg_sum_kW + P_vsg_kW
        Q_gen_total = Q_sg_sum_kvar + Q_vsg_kvar

        # Center of Inertia: только подключённые машины
        H_S_sum = 0.0
        H_S_omega_sum = 0.0
        for i_sg, sg_unit in enumerate(sg_list):
            sp = sg_params_list[i_sg]
            # Отключённые (Pe=0, P_set=0) не участвуют в CoI
            if sg_unit.Pe < 1e-6 and sg_unit.P_mech < 1e-6:
                continue
            HS = sp.H * sp.S_nom
            H_S_sum += HS
            H_S_omega_sum += HS * sg_unit.omega
        HS_vsg = vsg_params.H_virt * vsg_params.S_nom
        H_S_sum += HS_vsg
        H_S_omega_sum += HS_vsg * vsg.omega_virt
        omega_coi = H_S_omega_sum / (H_S_sum + 1e-9)

        f_bus, V_bus, P_load_actual, Q_load_actual = bus.step(
            dt, P_gen_total, Q_gen_total,
            P_load_base, Q_load_base,
            H_total, S_total,
            omega_coi,
            P_vsg_kW=P_vsg_kW, Q_vsg_kvar=Q_vsg_kvar,
            Xv_vsg=vsg_params.Xv
        )


        if k % ds == 0 and oi < n_out:
            out['t'][oi] = t
            out['f'][oi] = f_bus
            out['V'][oi] = V_bus
            out['P_sg_total'][oi] = P_sg_sum_kW
            out['Q_sg_total'][oi] = Q_sg_sum_kvar
            out['P_vsg'][oi] = P_vsg_kW
            out['Q_vsg'][oi] = Q_vsg_kvar
            out['P_load'][oi] = P_load_actual
            out['Q_load'][oi] = Q_load_actual
            out['SoC'][oi] = vsg.SoC
            out['omega_vsg'][oi] = vsg.omega_virt
            for i, sg in enumerate(sg_list):
                out['P_sg_each'][i, oi] = sg.Pe * sg.p.S_nom
                out['Q_sg_each'][i, oi] = sg.Qe * sg.p.S_nom
                out['P_mech'][i, oi] = sg.P_mech * sg.p.S_nom
                out['Efd'][i, oi] = sg.Efd
                out['Vt_sg'][i, oi] = sg.Vt
            oi += 1


    for key in out:
        if isinstance(out[key], np.ndarray):
            if out[key].ndim == 1:
                out[key] = out[key][:oi]
            else:
                out[key] = out[key][:, :oi]

    return out


def plot_results(res, n_sg, sg_p, vsg_p, save_path=None):
    t = res['t']
    fig = plt.figure(figsize=(16, 22))
    fig.suptitle(
        f'Параллельная работа {n_sg}xСГ ({sg_p.P_nom} кВт, H={sg_p.H}с) '
        f'+ ВСГ ({vsg_p.P_nom} кВт, H_virt={vsg_p.H_virt}с)\n'
        f'СГ: ramp^={sg_p.ramp_up * sg_p.P_nom:.0f} кВт/с, '
        f'droop={sg_p.R_droop * 100:.0f}%, T_gov={sg_p.T_gov}с  |  '
        f'ВСГ: D={vsg_p.D_virt}, droop={vsg_p.R_droop * 100:.0f}%, '
        f'K_rocof={vsg_p.K_rocof}, K_ff={vsg_p.K_ff}, '
        f'tau_inv={vsg_p.T_inv * 1000:.0f}мс, BESS={vsg_p.E_batt_kWh} кВт*ч',
        fontsize=12, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(6, 2, hspace=0.45, wspace=0.3,
                           left=0.07, right=0.96, top=0.94, bottom=0.03)

    C_SG = '#2166AC'
    C_VSG = '#1B7837'
    C_LOAD = '#D6604D'
    C_FREQ = '#7B3294'
    C_VOLT = '#E08214'
    C_sg_each = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#B2182B']


    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(t, 0, res['P_sg_total'], alpha=0.2, color=C_SG)
    ax.fill_between(t, res['P_sg_total'],
                    res['P_sg_total'] + np.maximum(res['P_vsg'], 0),
                    alpha=0.2, color=C_VSG)
    ax.plot(t, res['P_sg_total'], color=C_SG, lw=1.5, label='СГ суммарно')
    ax.plot(t, res['P_vsg'], color=C_VSG, lw=2, label='ВСГ (BESS)')
    ax.plot(t, res['P_load'], color=C_LOAD, lw=2, ls='--', label='Нагрузка')
    ax.plot(t, res['P_sg_total'] + res['P_vsg'], color='gray', lw=1,
            ls=':', alpha=0.6, label='Генерация суммарно')
    ax.set_ylabel('P, кВт')
    ax.set_title('Распределение активной мощности')
    ax.legend(loc='upper left', fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])


    ax = fig.add_subplot(gs[1, 0])
    for i in range(n_sg):
        c = C_sg_each[i % len(C_sg_each)]
        ax.plot(t, res['P_sg_each'][i], color=c, lw=1.5, label=f'Pe СГ-{i + 1}')
        ax.plot(t, res['P_mech'][i], color=c, lw=1, ls='--', alpha=0.6,
                label=f'Pm СГ-{i + 1}')
    ax.set_ylabel('P, кВт')
    ax.set_title('Мощности СГ (сплошная - электр., пунктир - мех.)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])


    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, res['P_vsg'], color=C_VSG, lw=2, label='P_ВСГ')
    ax.plot(t, res['Q_vsg'], color='#66C2A5', lw=1.5, ls='--', label='Q_ВСГ')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(vsg_p.P_nom, color=C_VSG, lw=0.5, ls=':', alpha=0.4)
    ax.axhline(-vsg_p.P_nom * 0.5, color='red', lw=0.5, ls=':', alpha=0.4)
    ax.set_ylabel('кВт / квар')
    ax.set_title('Мощность ВСГ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])


    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, res['f'], color=C_FREQ, lw=2)
    ax.axhline(50.0, color='gray', lw=0.5, ls='--')
    ax.fill_between(t, 49.8, 50.2, alpha=0.05, color='green')
    f_dev = np.abs(res['f'] - 50.0)
    idx = np.argmax(f_dev)
    ax.annotate(f'df_max = {f_dev[idx]:.3f} Гц',
                xy=(t[idx], res['f'][idx]),
                xytext=(t[idx] + 2, res['f'][idx] + 0.03 * np.sign(res['f'][idx] - 50)),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    ax.set_ylabel('f, Гц')
    ax.set_title('Частота шины')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])


    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, res['V'], color=C_VOLT, lw=2, label='V_шина')
    for i in range(min(n_sg, 3)):
        ax.plot(t, res['Vt_sg'][i], color=C_sg_each[i % len(C_sg_each)], lw=0.8, alpha=0.5,
                label=f'Vt СГ-{i + 1}')
    ax.axhline(1.0, color='gray', lw=0.5, ls='--')
    ax.fill_between(t, 0.95, 1.05, alpha=0.05, color='green')
    ax.set_ylabel('V, о.е.')
    ax.set_title('Напряжение (шина + клеммы СГ)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])


    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, res['SoC'] * 100, color='#1B9E77', lw=2)
    ax.axhline(vsg_p.SoC_min * 100, color='red', lw=1, ls='--',
               label=f'SoC_min={vsg_p.SoC_min * 100:.0f}%')
    ax.axhline(vsg_p.SoC_max * 100, color='blue', lw=1, ls='--',
               label=f'SoC_max={vsg_p.SoC_max * 100:.0f}%')
    ax.set_ylabel('SoC, %')
    ax.set_xlabel('Время, с')
    ax.set_title(f'Батарея ВСГ ({vsg_p.E_batt_kWh} кВт*ч)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 100)


    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, res['Q_sg_total'], color=C_SG, lw=1.5, label='Q_СГ')
    ax.plot(t, res['Q_vsg'], color=C_VSG, lw=1.5, label='Q_ВСГ')
    ax.plot(t, res['Q_load'], color=C_LOAD, lw=1.5, ls='--', label='Q_нагрузка')
    ax.set_ylabel('Q, квар')
    ax.set_xlabel('Время, с')
    ax.set_title('Реактивная мощность')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])


    ax = fig.add_subplot(gs[4, 0])
    for i in range(n_sg):
        ax.plot(t, res['Efd'][i], color=C_sg_each[i % len(C_sg_each)], lw=1.2,
                label=f'Efd СГ-{i + 1}')
    ax.set_ylabel('Efd, о.е.')
    ax.set_xlabel('Время, с')
    ax.set_title('Напряжение возбуждения (АРВ)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])


    ax = fig.add_subplot(gs[4, 1])
    f_vsg = res['omega_vsg'] / (2 * np.pi)
    ax.plot(t, res['f'], color=C_FREQ, lw=1.5, label='f_шина')
    ax.plot(t, f_vsg, color=C_VSG, lw=1.5, ls='--', label='f_ВСГ (виртуальная)')
    ax.axhline(50.0, color='gray', lw=0.5, ls='--')
    ax.set_ylabel('f, Гц')
    ax.set_xlabel('Время, с')
    ax.set_title('Частота шины vs виртуальная частота ВСГ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])


    ax = fig.add_subplot(gs[5, :])
    ax.plot(t, res['P_load'], color=C_LOAD, lw=2, label='P_нагрузка (с учётом f)')
    ax.plot(t, res['Q_load'], color='#FC8D62', lw=1.5, ls='--', label='Q_нагрузка')
    ax.set_ylabel('кВт / квар')
    ax.set_xlabel('Время, с')
    ax.set_title('Профиль нагрузки (с зависимостью P от частоты)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафики сохранены: {save_path}")
    return fig


def plot_transient_zoom(res, n_sg, sg_p, vsg_p, event_time=5.0, window=3.0, save_path=None):
    """
    Zoom-графики переходного процесса при набросе нагрузки.
    Показывает раскачку куста: перетоки P и Q между ВСГ и СГ,
    колебания V и f, реакцию АРВ.
    """
    t = res['t']
    # Окно: [event_time - 0.5, event_time + window]
    t0 = event_time - 0.5
    t1 = event_time + window
    mask = (t >= t0) & (t <= t1)
    tz = t[mask]

    C_SG = '#2166AC'
    C_VSG = '#1B7837'
    C_LOAD = '#D6604D'
    C_FREQ = '#7B3294'
    C_VOLT = '#E08214'

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(
        f'ZOOM: переходный процесс при набросе нагрузки (t = {event_time} с)\n'
        f'Взаимодействие ВСГ <-> куст СГ: перетоки P и Q, раскачка напряжения',
        fontsize=13, fontweight='bold'
    )

    ax = axes[0, 0]
    ax.plot(tz, res['P_sg_total'][mask], color=C_SG, lw=2, label='P СГ (сумма)')
    ax.plot(tz, res['P_vsg'][mask], color=C_VSG, lw=2, label='P ВСГ')
    ax.plot(tz, res['P_load'][mask], color=C_LOAD, lw=2, ls='--', label='P нагрузка')
    ax.plot(tz, res['P_sg_total'][mask] + res['P_vsg'][mask],
            color='gray', lw=1, ls=':', label='Генерация')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('P, кВт')
    ax.set_title('Активная мощность')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


    ax = axes[0, 1]
    ax.plot(tz, res['Q_sg_total'][mask], color=C_SG, lw=2, label='Q СГ (сумма)')
    ax.plot(tz, res['Q_vsg'][mask], color=C_VSG, lw=2, label='Q ВСГ')
    ax.plot(tz, res['Q_load'][mask], color=C_LOAD, lw=2, ls='--', label='Q нагрузка')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Q, квар')
    ax.set_title('Реактивная мощность - циркуляция Q')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


    ax = axes[1, 0]
    ax.plot(tz, res['V'][mask], color=C_VOLT, lw=2, label='V шина')
    ax.axhline(1.0, color='gray', lw=0.5, ls='--')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('V, о.е.')
    ax.set_title('Напряжение шины - толчок от ВСГ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


    ax = axes[1, 1]
    ax.plot(tz, res['f'][mask], color=C_FREQ, lw=2, label='f шина')
    f_vsg = res['omega_vsg'][mask] / (2 * np.pi)
    ax.plot(tz, f_vsg, color=C_VSG, lw=1.5, ls='--', label='f ВСГ (вирт.)')
    ax.axhline(50.0, color='gray', lw=0.5, ls='--')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('f, Гц')
    ax.set_title('Частота')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


    ax = axes[2, 0]
    ax.plot(tz, res['P_vsg'][mask], color=C_VSG, lw=2.5, label='P ВСГ')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    # Аннотация overshoot
    pv = res['P_vsg'][mask]
    idx_max = np.argmax(pv)
    ax.annotate(f'overshoot: {pv[idx_max]:.0f} кВт',
                xy=(tz[idx_max], pv[idx_max]),
                xytext=(tz[idx_max] + 0.3, pv[idx_max] * 0.9),
                fontsize=9, color=C_VSG,
                arrowprops=dict(arrowstyle='->', color=C_VSG, lw=0.8))
    ax.set_ylabel('P, кВт')
    ax.set_title('Мощность ВСГ - перерегулирование инвертора')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


    ax = axes[2, 1]
    for i in range(min(n_sg, 3)):
        ax.plot(tz, res['Efd'][i][mask], lw=1.5, label=f'Efd СГ-{i + 1}')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Efd, о.е.')
    ax.set_title('АРВ: реакция на возмущение V от ВСГ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


    ax = axes[3, 0]
    ax.plot(tz, res['P_sg_each'][0][mask], color=C_SG, lw=2, label='Pe СГ-1')
    ax.plot(tz, res['P_mech'][0][mask], color=C_SG, lw=1.5, ls='--', label='Pm СГ-1')
    ax.fill_between(tz, res['P_sg_each'][0][mask], res['P_mech'][0][mask],
                    alpha=0.15, color=C_SG)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('P, кВт')
    ax.set_xlabel('Время, с')
    ax.set_title('СГ-1: Pe vs Pm (заливка = dP -> изменение omega)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


    ax = axes[3, 1]
    P_gen_total = res['P_sg_total'][mask] + res['P_vsg'][mask]
    P_imb = P_gen_total - res['P_load'][mask]
    ax.plot(tz, P_imb, color='#E7298A', lw=2, label='P_gen − P_load')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.fill_between(tz, 0, P_imb, alpha=0.15, color='#E7298A')
    ax.set_ylabel('dP, кВт')
    ax.set_xlabel('Время, с')
    ax.set_title('Дисбаланс мощности на шине')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_all_machines_zoom(res, n_sg, sg_p, vsg_p, event_time=5.0, window=4.0, save_path=None):
    """
    Сводный zoom-график: все 9 СГ + ВСГ на одних осях.
    Показывает P и Q каждой машины при переходном процессе.
    """
    t = res['t']
    t0 = event_time - 0.5
    t1 = event_time + window
    mask = (t >= t0) & (t <= t1)
    tz = t[mask]

    C_VSG = '#1B7837'

    sg_colors = ['#2166AC', '#4393C3', '#92C5DE',
                 '#D1E5F0', '#F4A582', '#D6604D',
                 '#B2182B', '#762A83', '#5AAE61']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f'ВСЕ МАШИНЫ: переходный процесс при набросе нагрузки (t = {event_time} с)\n'
        f'9xСГ + 1xВСГ - распределение P и Q между генераторами',
        fontsize=13, fontweight='bold'
    )


    ax = axes[0, 0]
    for i in range(n_sg):
        ax.plot(tz, res['P_sg_each'][i][mask] / 1000, color=sg_colors[i % len(sg_colors)],
                lw=1.2, label=f'СГ-{i + 1}')
    ax.plot(tz, res['P_vsg'][mask] / 1000, color=C_VSG, lw=2.5, label='ВСГ',
            zorder=10)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('P, МВт')
    ax.set_title('Активная мощность каждой машины')
    ax.legend(fontsize=7, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)


    ax = axes[0, 1]
    for i in range(n_sg):
        ax.plot(tz, res['Q_sg_each'][i][mask] / 1000, color=sg_colors[i % len(sg_colors)],
                lw=1.2, label=f'СГ-{i + 1}')
    ax.plot(tz, res['Q_vsg'][mask] / 1000, color=C_VSG, lw=2.5, label='ВСГ',
            zorder=10)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Q, Мвар')
    ax.set_title('Реактивная мощность каждой машины')
    ax.legend(fontsize=7, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)


    ax = axes[1, 0]
    P_stack = np.zeros_like(tz)
    for i in range(n_sg):
        P_next = P_stack + res['P_sg_each'][i][mask] / 1000
        ax.fill_between(tz, P_stack, P_next, alpha=0.4,
                        color=sg_colors[i % len(sg_colors)], label=f'СГ-{i + 1}')
        P_stack = P_next

    P_vsg_mw = res['P_vsg'][mask] / 1000
    P_next = P_stack + np.maximum(P_vsg_mw, 0)
    ax.fill_between(tz, P_stack, P_next, alpha=0.6, color=C_VSG, label='ВСГ')
    ax.plot(tz, res['P_load'][mask] / 1000, color='red', lw=2, ls='--', label='Нагрузка')
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.3)
    ax.set_ylabel('P, МВт')
    ax.set_xlabel('Время, с')
    ax.set_title('Stacked: вклад каждой машины в покрытие нагрузки')
    ax.legend(fontsize=7, ncol=6, loc='upper left')
    ax.grid(True, alpha=0.3)


    ax = axes[1, 1]

    idx_base = np.searchsorted(tz, event_time - 0.1)
    if idx_base >= len(tz):
        idx_base = 0
    for i in range(n_sg):
        P_i = res['P_sg_each'][i][mask]
        dP_i = (P_i - P_i[idx_base]) / 1000
        ax.plot(tz, dP_i, color=sg_colors[i % len(sg_colors)], lw=1.2, label=f'dP СГ-{i + 1}')
    P_vsg_arr = res['P_vsg'][mask]
    dP_vsg = (P_vsg_arr - P_vsg_arr[idx_base]) / 1000
    ax.plot(tz, dP_vsg, color=C_VSG, lw=2.5, label='dP ВСГ', zorder=10)
    ax.axvline(event_time, color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('dP, МВт')
    ax.set_xlabel('Время, с')
    ax.set_title('Отклонение P от базы - кто и как реагирует на наброс')
    ax.legend(fontsize=7, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def _setup_sci_style():
    """Настройка matplotlib"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.4,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.4,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '0.85',
        'legend.fancybox': False,
    })


_C = {
    'sg': '#2166AC',
    'vsg': '#1B7837',
    'load': '#D6604D',
    'gen': '#878787',
    'freq': '#7B3294',
    'volt': '#E08214',
    'efd': '#B2182B',
    'imb': '#C51B7D',
    'fvsg': '#66C2A5',
    'qsg': '#4393C3',
    'qvsg': '#66C2A5',
    'qload': '#FC8D62',
}


def plot_sci_power_distribution(res, n_sg, sg_p, vsg_p, save_path=None):
    """Fig. 1: Распределение активной мощности (двойные оси Y)."""
    _setup_sci_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    t = res['t']
    P_sg = res['P_sg_total'] / 1000
    P_vsg = res['P_vsg'] / 1000
    P_load = res['P_load'] / 1000


    ax1.plot(t, P_sg, color=_C['sg'], lw=1.4, label=r'$P_{\mathrm{SG}}$ (9 ед.)')
    ax1.plot(t, P_load, color=_C['load'], lw=1.4, ls='--', label=r'$P_{\mathrm{load}}$')
    ax1.fill_between(t, 0, P_sg, alpha=0.08, color=_C['sg'])
    ax1.set_xlabel('Время, с')
    ax1.set_ylabel(r'$P_{\mathrm{SG}}$, $P_{\mathrm{load}}$, МВт', color=_C['sg'])
    ax1.tick_params(axis='y', labelcolor=_C['sg'])
    ax1.set_xlim(0, 60)

    ax1r = ax1.twinx()
    ax1r.plot(t, P_vsg, color=_C['vsg'], lw=1.6, label=r'$P_{\mathrm{VSG}}$')
    ax1r.fill_between(t, 0, P_vsg, alpha=0.10, color=_C['vsg'])
    ax1r.set_ylabel(r'$P_{\mathrm{VSG}}$, МВт', color=_C['vsg'])
    ax1r.tick_params(axis='y', labelcolor=_C['vsg'])
    ax1r.spines['right'].set_visible(True)
    ax1r.spines['right'].set_color(_C['vsg'])
    ax1r.axhline(0, color='0.8', lw=0.3, ls=':')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
    ax1.set_title('(a) Общий обзор', loc='left', fontsize=10)

    t0, t1 = 4.5, 9.0
    m = (t >= t0) & (t <= t1)

    ax2.plot(t[m], P_sg[m], color=_C['sg'], lw=1.6, label=r'$P_{\mathrm{SG}}$')
    ax2.plot(t[m], P_load[m], color=_C['load'], lw=1.4, ls='--', label=r'$P_{\mathrm{load}}$')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel(r'$P_{\mathrm{SG}}$, $P_{\mathrm{load}}$, МВт', color=_C['sg'])
    ax2.tick_params(axis='y', labelcolor=_C['sg'])
    ax2.axvline(5.0, color='0.6', lw=0.6, ls='--')

    ax2r = ax2.twinx()
    ax2r.plot(t[m], P_vsg[m], color=_C['vsg'], lw=1.8, label=r'$P_{\mathrm{VSG}}$')
    ax2r.fill_between(t[m], 0, P_vsg[m], alpha=0.12, color=_C['vsg'])
    ax2r.set_ylabel(r'$P_{\mathrm{VSG}}$, МВт', color=_C['vsg'])
    ax2r.tick_params(axis='y', labelcolor=_C['vsg'])
    ax2r.spines['right'].set_visible(True)
    ax2r.spines['right'].set_color(_C['vsg'])
    ax2r.axhline(0, color='0.8', lw=0.3, ls=':')

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
    ax2.set_title(r'(b) Zoom: наброс $t=5$ с', loc='left', fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_sci_power_distribution_single(res, n_sg, sg_p, vsg_p, save_path=None):
    """Fig. 1b: Распределение активной мощности (одна ось Y)."""
    _setup_sci_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    t = res['t']
    P_sg = res['P_sg_total'] / 1000
    P_vsg = res['P_vsg'] / 1000
    P_load = res['P_load'] / 1000
    P_gen = P_sg + P_vsg

    ax1.fill_between(t, 0, P_sg, alpha=0.12, color=_C['sg'])
    ax1.fill_between(t, P_sg, P_sg + np.maximum(P_vsg, 0), alpha=0.12, color=_C['vsg'])
    ax1.plot(t, P_sg, color=_C['sg'], lw=1.4, label=r'$P_{\mathrm{SG}}$ (9 ед.)')
    ax1.plot(t, P_vsg, color=_C['vsg'], lw=1.4, label=r'$P_{\mathrm{VSG}}$')
    ax1.plot(t, P_load, color=_C['load'], lw=1.4, ls='--', label=r'$P_{\mathrm{load}}$')
    ax1.plot(t, P_gen, color=_C['gen'], lw=0.9, ls=':', label=r'$\Sigma P_{\mathrm{gen}}$')
    ax1.set_xlabel('Время, с')
    ax1.set_ylabel(r'$P$, МВт')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 60)
    ax1.set_title('(a) Общий обзор', loc='left', fontsize=10)

    t0, t1 = 4.5, 9.0
    m = (t >= t0) & (t <= t1)
    ax2.plot(t[m], P_sg[m], color=_C['sg'], lw=1.6, label=r'$P_{\mathrm{SG}}$')
    ax2.plot(t[m], P_vsg[m], color=_C['vsg'], lw=1.6, label=r'$P_{\mathrm{VSG}}$')
    ax2.plot(t[m], P_load[m], color=_C['load'], lw=1.4, ls='--', label=r'$P_{\mathrm{load}}$')
    ax2.axvline(5.0, color='0.6', lw=0.6, ls='--')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel(r'$P$, МВт')
    ax2.legend(loc='upper right')
    ax2.set_title(r'(b) Zoom: наброс $t=5$ с', loc='left', fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_sci_frequency_voltage(res, n_sg, sg_p, vsg_p, save_path=None):
    """Fig. 2: Частота и напряжение шины."""
    _setup_sci_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8))

    t = res['t']
    f_bus = res['f']
    V_bus = res['V']
    f_vsg = res['omega_vsg'] / (2 * np.pi)

    ax = axes[0, 0]
    ax.plot(t, f_bus, color=_C['freq'], lw=1.2, label=r'$f_{\mathrm{bus}}$')
    ax.plot(t, f_vsg, color=_C['fvsg'], lw=1.0, ls='--', label=r'$f_{\mathrm{VSG}}$')
    ax.axhline(50.0, color='0.75', lw=0.5, ls=':')
    ax.set_ylabel(r'$f$, Гц')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 60)
    ax.set_title('(a) Частота шины', loc='left', fontsize=10)

    ax = axes[0, 1]
    ax.plot(t, V_bus, color=_C['volt'], lw=1.2, label=r'$V_{\mathrm{bus}}$')
    ax.axhline(1.0, color='0.75', lw=0.5, ls=':')
    ax.fill_between(t, 0.95, 1.05, alpha=0.05, color=_C['vsg'])
    ax.set_ylabel(r'$V$, о.е.')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 60)
    ax.set_title('(b) Напряжение шины', loc='left', fontsize=10)

    t0, t1 = 4.5, 9.0
    m = (t >= t0) & (t <= t1)

    ax = axes[1, 0]
    ax.plot(t[m], f_bus[m], color=_C['freq'], lw=1.5, label=r'$f_{\mathrm{bus}}$')
    ax.plot(t[m], f_vsg[m], color=_C['fvsg'], lw=1.2, ls='--', label=r'$f_{\mathrm{VSG}}$')
    ax.axhline(50.0, color='0.75', lw=0.5, ls=':')
    ax.axvline(5.0, color='0.6', lw=0.6, ls='--')
    f_dev = np.abs(f_bus[m] - 50.0)
    idx_max = np.argmax(f_dev)
    ax.annotate(f'$\\Delta f_{{\\max}}$ = {f_dev[idx_max]:.3f} Гц',
                xy=(t[m][idx_max], f_bus[m][idx_max]),
                xytext=(t[m][idx_max] + 0.5, f_bus[m][idx_max] + 0.02),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8, color=_C['freq']),
                color=_C['freq'])
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'$f$, Гц')
    ax.legend(loc='upper right')
    ax.set_title(r'(c) Zoom: частота при набросе', loc='left', fontsize=10)

    ax = axes[1, 1]
    ax.plot(t[m], V_bus[m], color=_C['volt'], lw=1.5, label=r'$V_{\mathrm{bus}}$')
    ax.axhline(1.0, color='0.75', lw=0.5, ls=':')
    ax.axvline(5.0, color='0.6', lw=0.6, ls='--')
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'$V$, о.е.')
    ax.legend(loc='upper right')
    ax.set_title(r'(d) Zoom: напряжение при набросе', loc='left', fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_sci_vsg_response(res, n_sg, sg_p, vsg_p, save_path=None):
    """Fig. 3: VSG response - P at load step + Q."""
    _setup_sci_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))

    t = res['t']
    t0, t1 = 4.5, min(12.0, t[-1])
    m = (t >= t0) & (t <= t1)


    ax = axes[0]
    pv = res['P_vsg'][m] / 1000
    ax.plot(t[m], pv, color=_C['vsg'], lw=1.6)
    ax.fill_between(t[m], 0, pv, alpha=0.12, color=_C['vsg'])
    ax.axhline(0, color='0.75', lw=0.4, ls=':')
    ax.axvline(5.0, color='0.6', lw=0.6, ls='--')
    if len(pv) > 0:
        idx_pk = np.argmax(pv)
        ax.annotate(f'{pv[idx_pk]:.2f} MW',
                    xy=(t[m][idx_pk], pv[idx_pk]),
                    xytext=(t[m][idx_pk] + 0.8, pv[idx_pk] * 0.80),
                    fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8, color=_C['vsg']),
                    color=_C['vsg'], fontweight='bold')
    ax.set_xlabel('Time, s')
    ax.set_ylabel(r'$P_{\mathrm{VSG}}$, MW')
    ax.set_title(r'(a) $P_{\mathrm{VSG}}$ at load step', loc='left', fontsize=9)


    ax = axes[1]
    ax.plot(t[m], res['Q_sg_total'][m] / 1000, color=_C['qsg'], lw=1.4, label=r'$Q_{\mathrm{SG}}$')
    ax.plot(t[m], res['Q_vsg'][m] / 1000, color=_C['qvsg'], lw=1.4, ls='--', label=r'$Q_{\mathrm{VSG}}$')
    ax.plot(t[m], res['Q_load'][m] / 1000, color=_C['qload'], lw=1.2, ls=':', label=r'$Q_{\mathrm{load}}$')
    ax.axvline(5.0, color='0.6', lw=0.6, ls='--')
    ax.set_xlabel('Time, s')
    ax.set_ylabel(r'$Q$, Mvar')
    ax.legend(loc='upper right')
    ax.set_title('(b) Reactive power', loc='left', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_sci_sg_dynamics(res, n_sg, sg_p, vsg_p, save_path=None):
    """Fig. 4: Динамика СГ - Pe/Pm, Efd, дисбаланс, губернатор."""
    _setup_sci_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8))

    t = res['t']
    t0, t1 = 4.5, 14.0
    m = (t >= t0) & (t <= t1)


    ax = axes[0, 0]
    ax.plot(t[m], res['P_sg_each'][0][m] / 1000, color=_C['sg'], lw=1.4, label=r'$P_{e}$ СГ-1')
    ax.plot(t[m], res['P_mech'][0][m] / 1000, color=_C['sg'], lw=1.2, ls='--', label=r'$P_{m}$ СГ-1')
    ax.fill_between(t[m], res['P_sg_each'][0][m] / 1000, res['P_mech'][0][m] / 1000,
                    alpha=0.12, color=_C['sg'])
    ax.axvline(5.0, color='0.6', lw=0.6, ls='--')
    ax.set_ylabel(r'$P$, МВт')
    ax.legend(loc='lower right')
    ax.set_title(r'(a) $P_e$ vs $P_m$ СГ-1', loc='left', fontsize=10)


    ax = axes[0, 1]
    ax.plot(t[m], res['Efd'][0][m], color=_C['efd'], lw=1.4, label=r'$E_{fd}$ СГ-1')
    ax.axvline(5.0, color='0.6', lw=0.6, ls='--')
    ax.set_ylabel(r'$E_{fd}$, о.е.')
    ax.legend(loc='upper right')
    ax.set_title('(b) Напряжение возбуждения', loc='left', fontsize=10)


    ax = axes[1, 0]
    P_gen = res['P_sg_total'][m] + res['P_vsg'][m]
    P_imb = (P_gen - res['P_load'][m]) / 1000
    ax.plot(t[m], P_imb, color=_C['imb'], lw=1.4, label=r'$\Delta P = P_{\mathrm{gen}} - P_{\mathrm{load}}$')
    ax.axhline(0, color='0.75', lw=0.5, ls=':')
    ax.axvline(5.0, color='0.6', lw=0.6, ls='--')
    ax.fill_between(t[m], 0, P_imb, where=(P_imb > 0), alpha=0.10, color=_C['vsg'])
    ax.fill_between(t[m], 0, P_imb, where=(P_imb < 0), alpha=0.10, color=_C['load'])
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'$\Delta P$, МВт')
    ax.legend(loc='upper right')
    ax.set_title('(c) Дисбаланс мощности', loc='left', fontsize=10)


    ax = axes[1, 1]
    ax.plot(t[m], res['P_mech'][0][m] / 1000, color=_C['sg'], lw=1.4, label=r'$P_{m}$ СГ-1')
    ax.plot(t[m], res['P_vsg'][m] / 1000, color=_C['vsg'], lw=1.4, ls='--', label=r'$P_{\mathrm{VSG}}$')
    ax.axvline(5.0, color='0.6', lw=0.6, ls='--')
    ax.axhline(0, color='0.75', lw=0.4, ls=':')
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'$P$, МВт')
    ax.legend(loc='right')
    ax.set_title(r'(d) Губернатор СГ vs ВСГ', loc='left', fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_prod_power_balance(res, n_sg, sg_p, vsg_p, save_path=None):
    """Power balance overview: P_SG_total, P_load, P_VSG, P_gen."""
    _setup_sci_style()
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    t = res['t']
    P_sg = res['P_sg_total'] / 1000
    P_vsg = res['P_vsg'] / 1000
    P_load = res['P_load'] / 1000
    P_gen = P_sg + P_vsg

    ax.plot(t, P_load, color=_C['load'], lw=1.8, ls='--', label=r'$P_{\mathrm{load}}$')
    ax.plot(t, P_gen, color=_C['gen'], lw=1.2, ls=':', label=r'$P_{\mathrm{gen}}$')
    ax.plot(t, P_sg, color=_C['sg'], lw=1.6, label=r'$\Sigma P_{\mathrm{SG}}$')
    ax.plot(t, P_vsg, color=_C['vsg'], lw=1.6, label=r'$P_{\mathrm{VSG}}$')

    ax.fill_between(t, 0, P_sg, alpha=0.10, color=_C['sg'])
    ax.fill_between(t, P_sg, P_sg + np.maximum(P_vsg, 0), alpha=0.15, color=_C['vsg'])

    ax.axvline(5.0, color='0.5', lw=0.6, ls='--')
    ax.set_xlabel('Time, s')
    ax.set_ylabel(r'$P$, MW')
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc='right', fontsize=8, ncol=1)
    ax.set_title('(a) Active power balance', loc='left', fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_prod_power_imbalance(res, n_sg, sg_p, vsg_p, save_path=None):
    """Power imbalance on the bus: dP = P_gen - P_load."""
    _setup_sci_style()
    fig, ax = plt.subplots(figsize=(7.16, 2.4))
    t = res['t']
    P_gen = (res['P_sg_total'] + res['P_vsg']) / 1000
    P_load = res['P_load'] / 1000
    dP = P_gen - P_load

    ax.plot(t, dP * 1000, color=_C['imb'], lw=1.4)
    ax.fill_between(t, 0, dP * 1000, where=(dP > 0), alpha=0.12, color=_C['vsg'])
    ax.fill_between(t, 0, dP * 1000, where=(dP < 0), alpha=0.12, color=_C['load'])
    ax.axhline(0, color='0.7', lw=0.5, ls=':')
    ax.axvline(5.0, color='0.5', lw=0.6, ls='--')

    ax.set_xlabel('Time, s')
    ax.set_ylabel(r'$\Delta P$, kW')
    ax.set_xlim(t[0], t[-1])
    ax.set_title('(b) Bus power imbalance', loc='left', fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_prod_individual_machines(res, n_sg, sg_p, vsg_p, save_path=None):
    """Individual machine powers: SG1, SG2, SG3, VSG."""
    _setup_sci_style()
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    t = res['t']

    sg_colors = ['#2166AC', '#4393C3', '#92C5DE']
    sg_labels = [r'$P_{\mathrm{SG1}}$', r'$P_{\mathrm{SG2}}$', r'$P_{\mathrm{SG3}}$']
    for i in range(min(n_sg, 3)):
        ax.plot(t, res['P_sg_each'][i] / 1000, color=sg_colors[i],
                lw=1.4, label=sg_labels[i])

    ax.plot(t, res['P_vsg'] / 1000, color=_C['vsg'], lw=1.8,
            label=r'$P_{\mathrm{VSG}}$', zorder=10)

    ax.axvline(5.0, color='0.5', lw=0.6, ls='--')
    ax.axhline(0, color='0.7', lw=0.4, ls=':')
    ax.set_xlabel('Time, s')
    ax.set_ylabel(r'$P$, MW')
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc='right', fontsize=8, ncol=1)
    ax.set_title('(c) Individual machine active power', loc='left', fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_prod_frequency_voltage(res, n_sg, sg_p, vsg_p, save_path=None):
    """Bus frequency, VSG virtual frequency, bus voltage."""
    _setup_sci_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 4.0), sharex=True)
    t = res['t']


    f_vsg = res['omega_vsg'] / (2 * np.pi)
    ax1.plot(t, res['f'], color=_C['freq'], lw=1.6, label=r'$f_{\mathrm{bus}}$')
    ax1.plot(t, f_vsg, color=_C['fvsg'], lw=1.2, ls='--', label=r'$f_{\mathrm{VSG}}$')
    ax1.axhline(50.0, color='0.7', lw=0.4, ls=':')
    ax1.axvline(5.0, color='0.5', lw=0.6, ls='--')
    ax1.set_ylabel(r'$f$, Hz')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('(d) Bus frequency and VSG virtual frequency', loc='left', fontsize=10)


    ax2.plot(t, res['V'], color=_C['volt'], lw=1.6, label=r'$V_{\mathrm{bus}}$')
    ax2.axhline(1.0, color='0.7', lw=0.4, ls=':')
    ax2.axvline(5.0, color='0.5', lw=0.6, ls='--')
    ax2.fill_between(t, 0.95, 1.05, alpha=0.04, color='green')
    ax2.set_xlabel('Time, s')
    ax2.set_ylabel(r'$V$, p.u.')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('(e) Bus voltage', loc='left', fontsize=10)
    ax2.set_xlim(t[0], t[-1])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ЗАПУСК

if __name__ == "__main__":
    print("=" * 70)
    print(" СЦЕНАРИЙ: 3xСГ + ВСГ - АСИММЕТРИЧНАЯ ЗАГРУЗКА")
    print(" СГ1,СГ2 = 95%, СГ3 = 0% (холостой ход на шине)")
    print(" Наброс +2.88 МВт -> ВСГ подхватывает -> раскачка при дележке")
    print("=" * 70)

    N_SG = 3

    sg_common = SGParams(
        P_nom=1600.0, S_nom=2000.0,
        P_max=0.80,
        H=3.5, D_mech=8.0,
        Xd=1.2, Xq=0.8, Xd_prime=0.30, Xd_dprime=0.18,
        Td0_prime=5.0, Td0_dprime=0.035,
        R_droop=0.025,
        T_act=0.05,
        T_gov=0.60,
        ramp_up=0.30, ramp_down=0.30,
        Ka=50.0, Ta=0.01, Te=1.5,
        Vref=1.0, Efd_max=5.0,
        X_line_sg=0.04, R_line_sg=0.005,
    )

    from copy import deepcopy

    sg_list_params = []
    for i in range(N_SG):
        sp = deepcopy(sg_common)
        sp.name = f"СГ-{i + 1}"
        sg_list_params.append(sp)


    P0_each = [0.95 * 1600.0 / 2000.0,  # СГ1: 0.76 о.е. = 1520 кВт
               0.95 * 1600.0 / 2000.0,  # СГ2: 0.76 о.е. = 1520 кВт
               0.0]  # СГ3: 0 (холостой ход)

    # Начальная нагрузка = P_sg1 + P_sg2 = 0.95*1600 + 0.95*1600 = 3040 кВт
    P0_load_kW = sum(P0_each[i] * sg_list_params[i].S_nom for i in range(N_SG))
    Q0_load_kvar = P0_load_kW * 0.3  # cosφ ≈ 0.96 - маленькая Q нагрузка


    P_step_kW = 560.0  # Наброс нагрузки
    P_after = P0_load_kW + P_step_kW  # 3600 кВт

    P_new_each = P_after / 3.0 / 2000.0  # 0.60 о.е. = 1200 кВт = 75% P_nom
    P_new_each = min(P_new_each, 0.80)

    P_set_schedule = [
        (5.0, [P_new_each, P_new_each, P_new_each]),
    ]


    vsg = VSGParams(
        P_nom=1600.0, S_nom=2000.0,
        P_max=0.80,
        H_virt=0.5, D_virt=15.0,
        R_droop=0.02,
        K_rocof=1.0, T_rocof=0.10,
        K_ff=1.0,
        T_inv=0.005, zeta_inv=0.65,
        I_max=1.2,
        Kp_v=20.0, Ki_v=300.0, T_qv=0.005,
        Xv=0.25,
        E_batt_kWh=1280.0,
        SoC_init=0.80,
        T_restore=0.0,
        T_washout=1.5,
    )

    bus = BusParams(
        f_nom=50.0, V_nom=1.0,
        X_line=0.08, R_line=0.008,
        D_load=1.0,
    )

    # ПРОФИЛЬ НАГРУЗКИ
    load = LoadProfile(events=[
        (0.0, P0_load_kW, Q0_load_kvar),  # 3040 кВт - начальная
        (5.0, P_after, P_after * 0.5),  # +1760 -> 4800 кВт - наброс
    ])

    sim = SimParams(dt=0.001, T_end=20.0, downsample=5)


    print(f"\nСуммарная установленная: {N_SG * 1600 / 1000:.1f} МВт СГ + "
          f"{vsg.P_nom / 1000:.1f} МВт ВСГ = {(N_SG * 1600 + vsg.P_nom) / 1000:.1f} МВт")
    print(f"\nНачальная загрузка:")
    for i, sp in enumerate(sg_list_params):
        p_kw = P0_each[i] * sp.S_nom
        print(f"  {sp.name}: P0 = {P0_each[i] * 100:.0f}% S_nom = {p_kw:.0f} кВт "
              f"({p_kw / sp.P_nom * 100:.0f}% P_nom)")
    print(f"  P0_load = {P0_load_kW:.0f} кВт")
    print(f"\nНаброс при t=5с: +{P_step_kW:.0f} кВт -> P_load = {P_after:.0f} кВт")
    print(f"  Новая уставка каждого СГ: {P_new_each:.3f} о.е. = {P_new_each * 2000:.0f} кВт "
          f"({P_new_each * 2000 / 1600 * 100:.0f}% P_nom)")
    print(f"  СГ1,2: СКИДЫВАЮТ с {P0_each[0] * 2000:.0f} -> {P_new_each * 2000:.0f} кВт")
    print(f"  СГ3:   НАБИРАЕТ  с 0 -> {P_new_each * 2000:.0f} кВт")
    print(f"  ВСГ: мгновенно компенсирует дефицит, потом droop")
    results = run_simulation(N_SG, sg_list_params, vsg, bus, load, sim,
                             P0_sg_pu=P0_each,
                             P_set_schedule=P_set_schedule)

    print(f"\n{'-' * 55}")
    print(f"РЕЗУЛЬТАТЫ:")
    print(f"  f:   {results['f'].min():.3f} .. {results['f'].max():.3f} Гц "
          f"(df_max = {np.abs(results['f'] - 50).max():.3f} Гц)")
    print(f"  V:   {results['V'].min():.4f} .. {results['V'].max():.4f} о.е.")
    print(f"  ВСГ: P_max = {results['P_vsg'].max():.0f} кВт "
          f"({results['P_vsg'].max() / vsg.P_nom * 100:.0f}%)")
    for i in range(N_SG):
        print(f"  {sg_list_params[i].name}: Pe_max = {results['P_sg_each'][i].max():.0f} кВт "
              f"({results['P_sg_each'][i].max() / sg_list_params[i].P_nom * 100:.0f}%)")
    print(f"  SoC: {results['SoC'][0] * 100:.1f}% -> {results['SoC'][-1] * 100:.1f}%")
    print(f"{'-' * 55}")

    sg_avg = sg_list_params[0]

    plot_results(results, N_SG, sg_avg, vsg, save_path="sg_vsg_results_v2.png")

    plot_transient_zoom(results, N_SG, sg_avg, vsg,
                        event_time=5.0, window=15.0,
                        save_path="zoom_transient_5s.png")

    plot_all_machines_zoom(results, N_SG, sg_avg, vsg,
                           event_time=5.0, window=15.0,
                           save_path="all_machines_5s.png")

    plot_sci_frequency_voltage(results, N_SG, sg_avg, vsg,
                               save_path="fig2_frequency_voltage.png")
    plot_sci_vsg_response(results, N_SG, sg_avg, vsg,
                          save_path="fig3_vsg_response.png")
    plot_sci_sg_dynamics(results, N_SG, sg_avg, vsg,
                         save_path="fig4_sg_dynamics.png")

    plot_prod_power_balance(results, N_SG, sg_avg, vsg,
                               save_path="prod_power_balance.png")
    plot_prod_power_imbalance(results, N_SG, sg_avg, vsg,
                                 save_path="prod_power_imbalance.png")
    plot_prod_individual_machines(results, N_SG, sg_avg, vsg,
                                     save_path="prod_individual_machines.png")
    plot_prod_frequency_voltage(results, N_SG, sg_avg, vsg,
                                   save_path="prod_freq_voltage.png")

    plt.close('all')
    print("\nГотово!")