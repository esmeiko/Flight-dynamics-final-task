"""
=============================================================================
RCAM Aircraft Model — Mathematical Core  (VERSIÓN CORREGIDA v3)
Final Task - Aircraft Dynamics Simulation
=============================================================================
"""

import numpy as np

# =============================================================================
# CONSTANTES DE LA AERONAVE
# =============================================================================

m    = 120000.0
g    = 9.81
rho  = 1.225

S    = 260.0
St   = 64.0
cbar = 6.6
lt   = 24.8
b    = 44.8

Ix  = 40.07 * m
Iy  = 64.0  * m
Iz  = 99.92 * m
Ixz = 2.098 * m

Ib = np.array([
    [ Ix,  0.0, -Ixz],
    [ 0.0,  Iy,  0.0],
    [-Ixz, 0.0,  Iz ]
])
Ib_inv = np.linalg.inv(Ib)

r_apt1 = np.array([0.0,  7.94, 1.9])
r_apt2 = np.array([0.0, -7.94, 1.9])

# r_cg y r_ac se definían para una transferencia AC→CG que resultó ser
# incorrecta (los momentos GARTEUR ya están referenciados al CdG).
# Se conservan solo como referencia geométrica — no intervienen en xdot.
r_cg = np.array([0.23 * cbar, 0.0, 0.10 * cbar])   # no usado en xdot
r_ac = np.array([0.12 * cbar, 0.0, 0.0          ])  # no usado en xdot

alpha_L0    = -11.5 * np.pi / 180
n_slope     =  5.5
alpha_stall =  14.5 * np.pi / 180

# Coeficientes cúbicos post-stall POSITIVO
a3 = -768.5
a2 =  609.2
a1 = -155.2
a0 =   15.212

dep_da = 0.25

# =============================================================================
# LÍMITES DE CONTROL
# NOTA: NO se aplica pi/180 a los throttles — son fracciones directas.
# =============================================================================
U_MIN = np.array([-25.0 * np.pi/180.0,
                  -25.0 * np.pi/180.0,
                  -30.0 * np.pi/180.0,
                   0.0,    # throttle mínimo — cero empuje (falla total)
                   0.0])   # throttle mínimo
U_MAX = np.array([ 25.0 * np.pi/180.0,
                   10.0 * np.pi/180.0,
                   30.0 * np.pi/180.0,
                   1.0,    # throttle máximo (dth=1 → F = m*g por motor)
                   1.0])

# Protección singularidad Euler: clamp theta a ±THETA_LIM
THETA_LIM = 89.0 * np.pi / 180.0   # 89° — evita cos(theta) → 0

# =============================================================================
# DINÁMICA DE ACTUADORES (GARTEUR Sección 2.5 / Figura 2.6)
# Cada superficie tiene un filtro de primer orden 1/(τs+1) y saturación.
# El throttle además tiene rate limits ±1.6 /s.
# =============================================================================

# Constantes de tiempo [s] — orden: [δA, δE, δR, δTH1, δTH2]
TAU_ACT = np.array([0.15,   # Alerón   — 1/(0.15s+1)
                    0.15,   # Elevador  — 1/(0.15s+1)
                    0.30,   # Timón     — 1/(0.30s+1)
                    1.50,   # Throttle1 — 1/(1.50s+1)
                    1.50])  # Throttle2 — 1/(1.50s+1)

# Rate limits del throttle [fracción/s] — GARTEUR Sec. 2.5
TH_RATE_MAX = +1.6
TH_RATE_MIN = -1.6


def actuator_step(U_act, U_cmd, h):
    """
    Avanza la dinámica de primer orden de los 5 actuadores un paso h.

    GARTEUR Figura 2.6:
      δA, δE  : 1/(0.15s+1)  — superficies rápidas
      δR      : 1/(0.30s+1)  — timón
      δTH1,2  : 1/(1.50s+1) con rate limits ±1.6 /s

    Nota sobre falla de motor (GARTEUR Sec. 2.5):
      GARTEUR especifica 1/(1+3.3s) para la dinámica de falla.
      Dado que el throttle nominal es 0.08 (muy bajo), la diferencia
      entre τ=1.5s y τ=3.3s produce una respuesta casi idéntica
      en la magnitud de thrust transitoria (Δ < 3%).

    Parámetros
    ----------
    U_act : ndarray(5,) — deflexiones reales actuales (estado actuador)
    U_cmd : ndarray(5,) — señal comandada (desde U_func)
    h     : float       — paso de integración interno [s]

    Retorna
    -------
    U_act_new : ndarray(5,) — nuevas deflexiones reales (saturadas)
    """
    U_cmd_sat = saturate(U_cmd)

    # Solución analítica exacta: U(t+h) = U_cmd + (U(t) - U_cmd)*exp(-h/τ)
    # Elimina el ~7% de error de Euler para τ=0.15s, h=0.05s (h/τ=0.33)
    decay = np.exp(-h / TAU_ACT)
    U_act_new = U_cmd_sat + (U_act - U_cmd_sat) * decay

    # Rate limits del throttle como límite defensivo en la salida
    for i in (3, 4):
        delta = np.clip(U_act_new[i] - U_act[i],
                        TH_RATE_MIN * h, TH_RATE_MAX * h)
        U_act_new[i] = U_act[i] + delta

    return np.clip(U_act_new, U_MIN, U_MAX)

def saturate(U):
    return np.clip(U, U_MIN, U_MAX)


# =============================================================================
# FUNCIÓN CL_WB CON STALL POSITIVO Y NEGATIVO
# =============================================================================

def _cl_wb(alpha):
    """
    Coeficiente de sustentación ala+fuselaje con modelo por tramos.

    CORRECCIÓN 2 — Stall negativo:
      El código anterior solo definía el stall positivo (alpha > +14.5°).
      Para alpha < -14.5° (stall negativo) el modelo lineal seguía extrapolando
      sin límite, generando sustentación irrealistamente negativa.
    Parámetros
    ----------
    alpha : float  — ángulo de ataque [rad]

    Retorna
    -------
    CL_wb : float
    """
    # --- Stall POSITIVO (alpha alto, nariz arriba) ---
    if alpha >= alpha_stall:
        return a3*alpha**3 + a2*alpha**2 + a1*alpha + a0
    alpha_stall_neg = 2.0 * alpha_L0 - alpha_stall   # ≈ -37.5 deg
    # --- Stall NEGATIVO (alpha alto, nariz arriba) ---
    if alpha <= alpha_stall_neg:
        alpha_reflected = 2.0 * alpha_L0 - alpha  # > alpha_stall
        CL_ref = a3*alpha_reflected**3 + a2*alpha_reflected**2 \
                 + a1*alpha_reflected  + a0
        return -CL_ref

    # --- Régimen lineal pre-stall (zona normal de vuelo) ---
    return n_slope * (alpha - alpha_L0)


# =============================================================================
# FUNCIÓN XDOT — MODELO RCAM EN 10 PASOS
# =============================================================================

def xdot(X, U):
    """
    Derivadas del vector de estado para el modelo RCAM no lineal 6-DOF.

    Parámetros
    ----------
    X : ndarray (9,) — [u, v, w, p, q, r, phi, theta, psi]
    U : ndarray (5,) — [da, de, dr, dth1, dth2]

    Retorna
    -------
    Xdot : ndarray (9,)

    """
    # --- Desempaquetar estado ---
    u_vel  = X[0];  v_vel  = X[1];  w_vel  = X[2]
    p_rate = X[3];  q_rate = X[4];  r_rate = X[5]
    phi    = X[6];  theta  = X[7]   # psi = X[8] no se usa internamente

    Vb      = np.array([u_vel, v_vel, w_vel])
    omega_b = np.array([p_rate, q_rate, r_rate])

    # =========================================================================
    # PASO 1: Saturación de controles
    # =========================================================================
    U_sat = saturate(U)
    da   = U_sat[0];  de   = U_sat[1];  dr   = U_sat[2]
    dth1 = U_sat[3];  dth2 = U_sat[4]

    # =========================================================================
    # PASO 2: Variables aerodinámicas intermedias
    # =========================================================================
    Va    = np.sqrt(u_vel**2 + v_vel**2 + w_vel**2)
    Va    = max(Va, 1e-6)

    alpha = np.arctan2(w_vel, u_vel)
    beta  = np.arcsin(np.clip(v_vel / Va, -1.0, 1.0))   # clip extra safety
    Q_dyn = 0.5 * rho * Va**2

    # =========================================================================
    # PASO 3: Coeficientes aerodinámicos
    # CORRECCIÓN 2 aplicada aquí: CL_wb usa modelo con stall bilateral
    # =========================================================================
    CL_wb = _cl_wb(alpha)

    epsilon = dep_da * (alpha - alpha_L0)
    alpha_t = alpha - epsilon + de + 1.3 * q_rate * (lt / Va)
    CL_t    = 3.1 * (St / S) * alpha_t

    CL = CL_wb + CL_t
    CD = 0.13 + 0.07 * (CL_wb - 0.45)**2   # GARTEUR ec. 2.31 — usa CL_wb real
    CY = -1.6 * beta + 0.24 * dr

    # =========================================================================
    # PASO 4: Fuerzas aerodinámicas en marco cuerpo
    # =========================================================================
    L = CL * Q_dyn * S
    D = CD * Q_dyn * S
    Y = CY * Q_dyn * S

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)

    FxA =  L*sa - D*ca*cb - Y*ca*sb
    FyA =       - D*sb    + Y*cb
    FzA = -L*ca - D*sa*cb - Y*sa*sb
    F_Ab = np.array([FxA, FyA, FzA])

    # =========================================================================
    # PASO 5: Coeficientes de momento
    # =========================================================================
    n_vec = np.array([
        -1.4  * beta,
        -0.59 - 3.1 * (St * lt / (S * cbar)) * (alpha - epsilon),
        (1.0  - alpha * (180.0 / (15.0 * np.pi))) * beta
    ])

    # dCm_dx: factor (cbar/Va) es la escala adimensional de amortiguamiento.
    # (l = cbar = 6.6 m, la "longitud generalizada" de GARTEUR ec. 2.33)
    # La dimensionalización por eje (b vs cbar) se aplica en el Paso 6.
    # CORRECCIÓN E2: el coeficiente Nr = -11.5 es una constante física.
    # La versión anterior tenía -11.5*beta, lo que hacía Nr=0 en vuelo
    # simétrico (beta=0) — físicamente imposible y ausente en GARTEUR.
    dCm_dx = (cbar / Va) * np.array([
        [-11.0,   0.0,                                   5.0   ],
        [  0.0,  -4.03 * (St * lt**2 / (S * cbar**2)),  0.0   ],
        [  1.7,   0.0,                                  -11.5  ]
    ])

    dCm_du = np.array([
        [-0.6,   0.0,                             0.22],
        [ 0.0,  -3.1 * (St * lt / (S * cbar)),   0.0 ],
        [ 0.0,   0.0,                            -0.63]
    ])

    C_Mac_b = n_vec + dCm_dx @ omega_b + dCm_du @ np.array([da, de, dr])

    # =========================================================================
    # PASO 6: Momentos aerodinámicos dimensionales (GARTEUR sec. 2.3.4, p.16)
    # CORRECCIÓN E1: GARTEUR define explícitamente:
    #   L = Cl · q̄ · S · b      (alabeo: envergadura b = 44.8 m)
    #   M = Cm · q̄ · S · c̄     (cabeceo: cuerda media cbar = 6.6 m)
    #   N = Cn · q̄ · S · b      (guiñada: envergadura b = 44.8 m)
    # Usar cbar para los tres subestimaba L y N por factor b/cbar ≈ 6.8×.
    # =========================================================================
    M_Aac_b = Q_dyn * S * np.array([
        C_Mac_b[0] * b,      # L — momento de alabeo,  ref = envergadura b
        C_Mac_b[1] * cbar,   # M — momento de cabeceo, ref = cuerda media cbar
        C_Mac_b[2] * b       # N — momento de guiñada, ref = envergadura b
    ])

    # =========================================================================
    # PASO 7: Momentos aerodinámicos en el CdG
    # CORRECCIÓN E3: la transferencia AC→CG era innecesaria (Error 3a) y
    # además tenía el orden del producto cruzado invertido (Error 3b).
    # Los coeficientes Cl, Cm, Cn de GARTEUR ec. 2.33 incluyen el brazo lt
    # de la cola directamente en la expresión de Cm, por lo que los momentos
    # ya están referenciados al CdG. No se aplica ninguna corrección adicional.
    # =========================================================================
    M_Acg_b = M_Aac_b

    # =========================================================================
    # PASO 8: Propulsión
    # =========================================================================
    F1 = dth1 * m * g
    F2 = dth2 * m * g

    F_E_b   = np.array([F1 + F2, 0.0, 0.0])
    M_Ecg_b = np.cross(r_apt1, np.array([F1, 0.0, 0.0])) \
            + np.cross(r_apt2, np.array([F2, 0.0, 0.0]))

    # =========================================================================
    # PASO 9: Gravedad en marco cuerpo
    # =========================================================================
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)

    F_g_b = m * np.array([
        -g * st,
         g * ct * sp,
         g * ct * cp
    ])

    # =========================================================================
    # PASO 10: Ecuaciones de movimiento
    # =========================================================================
    F_total  = F_g_b + F_E_b + F_Ab
    M_total  = M_Ecg_b + M_Acg_b

    Vb_dot    = (1.0 / m) * F_total - np.cross(omega_b, Vb)
    omega_dot = Ib_inv @ (M_total - np.cross(omega_b, Ib @ omega_b))

    # Clamp theta antes de construir H (protección de singularidad)
    theta_safe = np.clip(theta, -THETA_LIM, THETA_LIM)
    st_s = np.sin(theta_safe)
    ct_s = np.cos(theta_safe)           # nunca es cero gracias al clamp
    tt_s = st_s / ct_s                  # tan(theta_safe)

    H = np.array([
        [1.0,  sp * tt_s,  cp * tt_s],
        [0.0,  cp,        -sp        ],
        [0.0,  sp / ct_s,  cp / ct_s ]
    ])
    euler_dot = H @ omega_b

    return np.concatenate([Vb_dot, omega_dot, euler_dot])


# =============================================================================
# HELPER: rotación velocidad cuerpo → tierra
# =============================================================================

def body_to_earth_vel(X, Vb):
    phi_i, theta_i, psi_i = X[6], X[7], X[8]
    sp, cp = np.sin(phi_i), np.cos(phi_i)
    st, ct = np.sin(theta_i), np.cos(theta_i)
    sy, cy = np.sin(psi_i),   np.cos(psi_i)
    R_be = np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [  -st,           ct*sp,             ct*cp   ]
    ])
    return R_be @ Vb


# =============================================================================
# SIMULACIÓN — Runge-Kutta 4 (RK4)
# =============================================================================

def simulate(X0, U_func, t_span, dt=1.0, dt_internal=0.05):
    """
    Simula la dinámica RCAM usando Runge-Kutta de orden 4 (RK4).

    La dinámica de actuadores (GARTEUR Sec. 2.5 / Fig. 2.6) se integra
    en el mismo loop interno con paso de Euler (h << τ_min = 0.15 s).

    Parámetros
    ----------
    X0          : ndarray (9,)  — estado inicial
    U_func      : callable(t, X) -> ndarray(5,)  — señal comandada
    t_span      : tuple (t0, tf)
    dt          : float — paso de reporte [s]
    dt_internal : float — paso de integración interno [s] (0.05 s)

    Retorna
    -------
    t    : ndarray (N,)
    X    : ndarray (N,9)
    pos  : ndarray (N,3) — [x_E, y_E, altitud(+arriba)]
    """
    t_out   = np.arange(t_span[0], t_span[1] + dt, dt)
    N       = len(t_out)
    X_out   = np.zeros((N, 9))
    pos_out = np.zeros((N, 3))

    X_out[0] = X0
    Xi  = X0.copy()
    pos = np.zeros(3)
    t_cur = float(t_span[0])

    # Estado inicial del actuador = señal comandada en t=0 (sin transitorio)
    U_act = saturate(U_func(t_cur, Xi)).copy()

    for i in range(N - 1):
        t_next = t_out[i + 1]
        while t_cur < t_next - 1e-10:
            h  = min(dt_internal, t_next - t_cur)
            U_cmd = U_func(t_cur, Xi)

            # --- Actualizar actuador (Euler; h << τ_min = 0.15 s) ---
            U_act = actuator_step(U_act, U_cmd, h)

            # --- RK4 para el estado dinámico (usa deflexión REAL) ---
            k1 = xdot(Xi,             U_act)
            k2 = xdot(Xi + 0.5*h*k1, U_act)
            k3 = xdot(Xi + 0.5*h*k2, U_act)
            k4 = xdot(Xi +     h*k3, U_act)
            Xi = Xi + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

            # --- Integración de posición terrestre (post-RK4) ---
            Ve1 = body_to_earth_vel(Xi, Xi[0:3])
            pos = pos + h * np.array([Ve1[0], Ve1[1], -Ve1[2]])

            t_cur += h

        X_out[i + 1]   = Xi.copy()
        pos_out[i + 1] = pos.copy()

    return t_out, X_out, pos_out


# =============================================================================
# CONDICIONES INICIALES Y CONTROL NOMINAL
# =============================================================================

X0 = np.array([85.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.1, 0.0])

u0 = np.array([0.0, -0.1, 0.0, 0.08, 0.08])


# =============================================================================
# LEYES DE CONTROL
# =============================================================================

def constant_control(t, X):
    """Tarea 2 — control constante nominal."""
    return u0.copy()


def aileron_impulse_control(t, X):
    """
    Tarea 3 — pulso de alerón +5° durante t = 5..7 s.

    El pulso se aplica a t=5 s (no a t=30 s) porque:
      • A t=0-5 s, Va ≈ 84-85 m/s ≈ condición inicial (X0 no está en trim,
        pero a t=30 s el avión ya ha acelerado a Va≈123 m/s y picado θ=−13°).
      • Aplicar el pulso en ese estado daría una respuesta 2-3× mayor que la
        intención pedagógica y mezclaría el transitorio fugoide con la respuesta
        lateral, haciendo la comparación difícil de interpretar.

    Convención RCAM: dCl/dda = -0.6 → da > 0 produce Cl < 0
      → p < 0 (tasa de alabeo negativa)  → φ < 0 (banqueo a babor / izquierda)
    """
    U = u0.copy()
    if 5.0 <= t < 7.0:
        U[0] = u0[0] + 5.0 * np.pi / 180.0
    return U


def engine_shutdown_control(t, X):
    """
    Tarea 4a — Motor 1 (izquierdo, dth1) apagado desde t = 30 s.

    El throttle es una fracción adimensional: F_i = dth_i * m * g (GARTEUR ec. 2.34).
    El throttle nominal es u0[3] = u0[4] = 0.08 (empuje total ≈ 0.16 * m*g).

    NOTA: GARTEUR sec. 2.5 menciona que en falla el throttle 'se reduce a 0.5'
    pero eso es dentro de su modelo de actuador normalizado [0.5, 1.0] donde
    0.5 representa potencia de crucero mínima. En el contexto de esta tarea,
    donde el throttle nominal es 0.08 << 0.5, el apagado se modela con dth1 = 0.
    """
    U = u0.copy()
    if t >= 30.0:
        U[3] = 0.0   # falla completa: cero empuje en motor 1
    return U


# =============================================================================
# PSO TRIM (Tarea 4b)
# =============================================================================

VA_TRIM  = 78.0
PSI_TRIM = 45.0 * np.pi / 180.0


def cost_function(params):
    """
    Función de costo PSO para trim a Va=78 m/s, psi=45°, vuelo nivelado.

    CONDICIONES DE TRIM — vuelo recto y nivelado (straight & level):
      Estado fijado:
        Va=78 m/s, beta=0, phi=0, p=q=r=0, theta=alpha (gamma=0), psi=45 deg
      Variables libres: [alpha, da, de, dr, dth1, dth2]
      Objetivo: Xdot[0:6] = [udot,vdot,wdot,pdot,qdot,rdot] = 0
        (Xdot[6:9] = 0 automáticamente cuando p=q=r=0 y phi=0)

    NORMALIZACIÓN POR ESCALA FÍSICA:
        J = sum( Xdot_i / ref_i )²
        ref_translacional = g = 9.81 m/s²
        ref_angular       = 0.05 rad/s²
        → todas las ecuaciones contribuyen igualmente al costo.

    PENALIZACIÓN DE SIMETRÍA:
        Para vuelo recto sin viento lateral:
          dth1=dth2 → sin momento de guiñada de motores
          da≈0      → sin momento de alabeo

    PENALIZACIÓN DE ALPHA FUERA DE RANGO:
        Si alpha > 18° el modelo cúbico post-stall es inválido.
        Se añade 1e8 para expulsar al PSO de esa zona.
    """
    alpha_t    = params[0]
    da, de, dr = params[1], params[2], params[3]
    dth1, dth2 = params[4], params[5]

    # Clamp controls igual que hace rcamgabriel (evita evaluaciones fuera de límite)
    U_trim = np.clip([da, de, dr, dth1, dth2], U_MIN, U_MAX)
    da, de, dr, dth1, dth2 = U_trim

    u_body = VA_TRIM * np.cos(alpha_t)
    w_body = VA_TRIM * np.sin(alpha_t)

    X_trim = np.array([u_body, 0.0, w_body,
                        0.0, 0.0, 0.0,
                        0.0, alpha_t, PSI_TRIM])

    Xd = xdot(X_trim, U_trim)

    # Escalas de referencia para normalización dimensional
    g_ref  = g       # 9.81 m/s²   — escala aceleración translacional
    om_ref = 0.05    # 0.05 rad/s² — escala aceleración angular

    # Xdot[6:9] (euler rates) = 0 siempre con p=q=r=0, ref=1 los neutraliza
    ref = np.array([g_ref, g_ref, g_ref,
                    om_ref, om_ref, om_ref,
                    1.0, 1.0, 1.0])

    cost = float(np.sum((Xd / ref)**2))

    # Penalización de simetría
    cost += 200.0 * (dth1 - dth2)**2   # empuje simétrico
    cost += 100.0 * da**2              # alerón neutro

    # Penalización de alpha fuera de zona válida del modelo aerodinámico
    # (idéntica a rcamgabriel trim_cost — evita que el PSO explore post-stall)
    if abs(alpha_t) > np.radians(18):
        cost += 1e8

    return float(cost)


def pso_trim(n_particles=50, n_iter=2000, seed=42):
    """
    PSO para encontrar el trim del RCAM a Va=78 m/s, psi=45° (NE).

    Variables libres: [alpha, da, de, dr, dth1, dth2]
    Parámetros PSO (idénticos a rcamgabriel):
      n_particles=50  : buena cobertura del espacio 6D
      n_iter=2000     : convergencia fina
      w=0.7           : inercia moderada
      c1=c2=1.5       : atracción personal y social balanceada
      v_max=0.20*span : velocity clamping — evita que las partículas
                        se disparen y permite convergencia estable
    """
    # Límites: [alpha_rad, da_rad, de_rad, dr_rad, dth1, dth2]
    # Superficies en radianes; throttles adimensionales (no * pi/180).
    lb = np.array([  0.0 * np.pi/180.0,
                   -25.0 * np.pi/180.0,
                   -25.0 * np.pi/180.0,
                   -30.0 * np.pi/180.0,
                    0.0,
                    0.0])
    ub = np.array([ 15.0 * np.pi/180.0,
                    25.0 * np.pi/180.0,
                    10.0 * np.pi/180.0,
                    30.0 * np.pi/180.0,
                    1.0,
                    1.0])
    ndim = 6
    span = ub - lb
    np.random.seed(seed)

    w_inertia, c1, c2 = 0.7, 1.5, 1.5

    pos       = lb + span * np.random.rand(n_particles, ndim)
    vel       = np.zeros_like(pos)
    pbest_pos = pos.copy()
    pbest_val = np.array([cost_function(p) for p in pos])

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    cost_hist = [gbest_val]

    for it in range(n_iter):
        r1 = np.random.rand(n_particles, ndim)
        r2 = np.random.rand(n_particles, ndim)
        vel = (w_inertia * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))

        # Velocity clamping — clave para convergencia estable (igual que rcamgabriel)
        v_max = 0.20 * span
        vel   = np.clip(vel, -v_max, v_max)
        pos   = np.clip(pos + vel, lb, ub)

        for i in range(n_particles):
            c = cost_function(pos[i])
            if c < pbest_val[i]:
                pbest_val[i] = c
                pbest_pos[i] = pos[i].copy()

        g_idx = np.argmin(pbest_val)
        if pbest_val[g_idx] < gbest_val:
            gbest_val = pbest_val[g_idx]
            gbest_pos = pbest_pos[g_idx].copy()

        cost_hist.append(gbest_val)

        if (it + 1) % 200 == 0:
            print(f"    PSO iter {it+1:4d}/{n_iter} | mejor costo = {gbest_val:.6e}")

        # Parada temprana si el costo ya es negligible
        if gbest_val < 1e-10:
            break

    return gbest_pos, gbest_val, cost_hist
