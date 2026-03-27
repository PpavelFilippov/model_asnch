"""
N-фазная математическая модель
Источники:
- Глазырин А.С.
Решатель: LSODA
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve as la_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Параметры двигателя АДМ100S4У3
R1     = 1.851      # Ом
R2     = 2.236      # Ом
LM_3PH = 0.2138     # Гн
L1_SIG = 0.011      # Гн
L2_SIG = 0.014      # Гн
ZP     = 2
N_BARS = 28
F_NET  = 50         # Гц
U_PH   = 220        # В
JE     = 0.01       # кг*м^2
MN     = 20.3       # Н*м


class InductionMotorNPhase:

    def __init__(self, n=28, zp=2, Rs=1.851, Rr=2.236,
                 Ls_sigma=0.011, Lr_sigma=0.014, Lm_3ph=0.2138,
                 J=0.01, Uph=220, f=50,
                 faulty_bars=None, R_fault=1e6):

        self.n   = n
        self.zp  = zp
        self.J   = J
        self.Uph = Uph
        self.f   = f
        self.w   = 2.0 * np.pi * f
        self.Um  = np.sqrt(2.0) * Uph

        # Приведённая Lm: Lm1 = Lm_3ph * 2 / n
        self.Lm = Lm_3ph * 2.0 / n

        # Полные индуктивности ( L1 = 0.011 + Lm; L2 = 0.014 + Lm)
        self.Ls = Ls_sigma + self.Lm
        self.Lr = Lr_sigma + self.Lm

        self.phi = 2.0 * np.pi / n

        # Сопротивления
        self.Rs_vec = np.full(n, Rs)
        self.Rr_vec = np.full(n, Rr)
        if faulty_bars is not None:
            for idx in faulty_bars:
                self.Rr_vec[idx] += R_fault

        # Corr(phi): Corr(i,j) = cos((j−i)*phi), диагональ = 0
        idx_diff = np.arange(n)[:, None] - np.arange(n)[None, :]
        Corr = np.cos(idx_diff * self.phi)
        np.fill_diagonal(Corr, 0.0)

        # Ls0, Lr0 - постоянные матрицы
        self.Ls0 = self.Lm * Corr + np.diag(np.full(n, self.Ls))
        self.Lr0 = self.Lm * Corr + np.diag(np.full(n, self.Lr))

        # Индексная сетка для Cosr  (i+j)*phi
        self._ij_phi = (np.arange(n)[:, None] + np.arange(n)[None, :]) * self.phi

        # Углы для n-фазных напряжений
        self._phase_offsets = np.arange(n) * self.phi

        self.state_size = 2 * n + 2

    # Cosr-матрица (блок 4): Cosr(i,j)=cos(γ − j*phi − i*phi)
    def _cosr(self, ge):
        return np.cos(ge - self._ij_phi)

    # Sin-матрица (блок 5): m(i,j)=sin(γ − j*phi − i*phi)
    def _sinr(self, ge):
        return np.sin(ge - self._ij_phi)

    # L0 (полная матрица индуктивностей 2n×2n)
    def _build_L0(self, ge):
        n = self.n
        Cosr = self._cosr(ge)
        Lrs = self.Lm * Cosr              # n×n
        L0 = np.empty((2*n, 2*n))
        L0[:n, :n] = self.Ls0
        L0[:n, n:] = Lrs.T                # Lsr = Lrs^T
        L0[n:, :n] = Lrs
        L0[n:, n:] = self.Lr0
        return L0

    # Напряжения (блок 2): 3-фаз -> n-фаз
    def _voltages(self, t):
        wt = self.w * t
        Ua = self.Um * np.sin(wt)
        Ub = self.Um * np.sin(wt - 2.0*np.pi/3.0)
        Uc = self.Um * np.sin(wt + 2.0*np.pi/3.0)
        dbc = Ub - Uc
        Uabs = np.sqrt(Ua*Ua + dbc*dbc / 3.0)
        alpha = np.arctan2(Ua, dbc / np.sqrt(3.0))
        return Uabs * np.cos(alpha + self._phase_offsets)

    # Момент (блок 5): Te = −zp*Is^T*(Lm*m)*Ir*3/n
    def _torque(self, Is, Ir, ge):
        m = self._sinr(ge)
        return -self.zp * (Is @ (self.Lm * m) @ Ir) * 3.0 / self.n

    # Огибающая: Im = sqrt(ΣIs^2)*sqrt(3/n)
    def envelope(self, Is):
        return np.sqrt(np.dot(Is, Is) * 3.0 / self.n)

    # Правая часть ОДУ
    def rhs(self, t, y, Mc_func):
        n = self.n
        psi     = y[:2*n]
        omega   = y[2*n]
        gamma_e = y[2*n+1]

        L0 = self._build_L0(gamma_e)
        I_full = la_solve(L0, psi, overwrite_a=True, check_finite=False)
        Is = I_full[:n]
        Ir = I_full[n:]

        Us = self._voltages(t)
        Te = self._torque(Is, Ir, gamma_e)

        dy = np.empty(2*n+2)
        dy[:n]     = Us - self.Rs_vec * Is
        dy[n:2*n]  = -self.Rr_vec * Ir
        dy[2*n]    = (Te - Mc_func(t)) / self.J
        dy[2*n+1]  = self.zp * omega
        return dy

    # Запуск
    def simulate(self, t_span, Mc_func=None, y0=None,
                 dt_out=0.0005, max_step=0.0002,
                 rtol=1e-6, atol=1e-8):
        if y0 is None:
            y0 = np.zeros(self.state_size)
        if Mc_func is None:
            Mc_func = lambda t: 0.0

        t_eval = np.arange(t_span[0], t_span[1] + dt_out*0.5, dt_out)

        print(f"  n={self.n}, dim={self.state_size}, "
              f"t=[{t_span[0]:.2f}..{t_span[1]:.2f}]с, max_step={max_step}")

        t0 = time.time()
        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, Mc_func),
            t_span=t_span, y0=y0, method='LSODA',
            t_eval=t_eval, max_step=max_step,
            rtol=rtol, atol=atol,
        )
        elapsed = time.time() - t0
        s = "OK" if sol.success else f"FAIL: {sol.message}"
        print(f" {s}, {len(sol.t)} точек, {elapsed:.1f} с")
        return sol

    # Постобработка
    def postprocess(self, sol):
        n = self.n
        t = sol.t
        Np = len(t)
        omega   = sol.y[2*n]
        gamma_e = sol.y[2*n+1]

        Te    = np.empty(Np)
        I_env = np.empty(Np)
        Is    = np.empty((n, Np))
        Ir    = np.empty((n, Np))

        for k in range(Np):
            psi = sol.y[:2*n, k]
            ge  = gamma_e[k]
            L0  = self._build_L0(ge)
            I_full = la_solve(L0, psi, check_finite=False)
            Is[:, k] = I_full[:n]
            Ir[:, k] = I_full[n:]
            Te[k]    = self._torque(I_full[:n], I_full[n:], ge)
            I_env[k] = self.envelope(I_full[:n])

        return dict(t=t, omega=omega, gamma_e=gamma_e,
                    Te=Te, I_env=I_env, Is=Is, Ir=Ir)


def make_load_func(t_load, Mn):
    def f(t):
        return Mn if t >= t_load else 0.0
    return f


def quick_test():
    """Короткий прогон 0–0.1 с"""
    print("=" * 60)
    print("БЫСТРЫЙ ТЕСТ (0–0.1 с)")
    print("=" * 60)

    motor = InductionMotorNPhase()
    sol = motor.simulate(t_span=(0, 0.1), dt_out=0.001, max_step=0.0002)

    if not sol.success:
        print("ТЕСТ НЕ ПРОШЁЛ!")
        return False

    omega_end = sol.y[2*N_BARS, -1]
    print(f"  omega(0.1с) = {omega_end:.2f} рад/с - {'OK' if omega_end > 0 else 'ПРОБЛЕМА'}")

    psi = sol.y[:2*N_BARS, -1]
    ge  = sol.y[2*N_BARS+1, -1]
    L0  = motor._build_L0(ge)
    I   = la_solve(L0, psi, check_finite=False)
    Ienv = motor.envelope(I[:N_BARS])
    print(f"  I_огиб(0.1с) = {Ienv:.2f} А - {'OK' if np.isfinite(Ienv) else 'ПРОБЛЕМА'}")
    return True


def full_simulation():
    print("\n" + "=" * 60)
    print("ПОЛНОЕ МОДЕЛИРОВАНИЕ (0–3 с)")
    print("=" * 60)

    Mc_func = make_load_func(t_load=2.0, Mn=MN)
    results = {}

    configs = [
        ("Без обрыва",       None),
        ("Обрыв 1 стержня",  [0]),
        ("Обрыв 3 стержней", [0, 1, 2]),
    ]

    for label, faults in configs:
        print(f"\n {label} ")
        motor = InductionMotorNPhase(faulty_bars=faults)
        sol = motor.simulate(t_span=(0, 3.0), Mc_func=Mc_func,
                             dt_out=0.0005, max_step=0.0002)
        if sol.success:
            results[label] = motor.postprocess(sol)
        else:
            print(f"  !!! {sol.message}")

    return results


def plot_results(results):
    if not results:
        return

    colors = {"Без обрыва": 'tab:blue',
              "Обрыв 1 стержня": 'tab:green',
              "Обрыв 3 стержней": 'tab:red'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for label, res in results.items():
        c = colors.get(label, 'k')
        axes[0].plot(res['t'], res['Te'], color=c, lw=0.5, label=label, alpha=0.8)
        axes[1].plot(res['t'], res['omega'], color=c, lw=1, label=label)
        axes[2].plot(res['t'], res['I_env'], color=c, lw=0.7, label=label)

    axes[0].set_ylabel('M, Н*м'); axes[0].set_title('Электромагнитный момент')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim([-15, 70])
    axes[1].set_ylabel('omega, рад/с'); axes[1].set_title('Угловая скорость ротора')
    axes[1].legend(loc='lower right'); axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel('I, А'); axes[2].set_xlabel('t, с')
    axes[2].set_title('Огибающая тока статора')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ad_nphase_main.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ad_nphase_main.png")

    # Детальные фрагменты
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for label, res in results.items():
        c = colors.get(label, 'k')
        t = res['t']
        m1 = (t >= 1.0) & (t <= 2.0)
        axes[0].plot(t[m1], res['I_env'][m1], color=c, lw=1, label=label)
        m2 = (t >= 2.0) & (t <= 3.0)
        axes[1].plot(t[m2], res['I_env'][m2], color=c, lw=1, label=label)

    axes[0].set_title('Огибающая - холостой ход (1–2 с)')
    axes[0].set_ylabel('I, А'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Огибающая - нагрузка (2–3 с)')
    axes[1].set_ylabel('I, А'); axes[1].set_xlabel('t, с')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ad_nphase_envelope.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ad_nphase_envelope.png")


if __name__ == '__main__':
    if not quick_test():
        raise SystemExit("Быстрый тест не прошёл.")

    results = full_simulation()

    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    for label, res in results.items():
        t = res['t']
        m1 = (t >= 1.5) & (t <= 2.0)
        m2 = (t >= 2.5) & (t <= 3.0)
        print(f"  {label}:")
        if m1.any():
            w = np.mean(res['omega'][m1])
            print(f"    ХХ:    omega={w:.1f} рад/с ({w*30/np.pi:.0f} об/мин), "
                  f"I={np.mean(res['I_env'][m1]):.3f} А")
        if m2.any():
            w = np.mean(res['omega'][m2])
            print(f"    Нагр.: omega={w:.1f} рад/с ({w*30/np.pi:.0f} об/мин), "
                  f"I={np.mean(res['I_env'][m2]):.3f} А")

    print("\nПостроение графиков...")
    plot_results(results)
    print("Готово.")
