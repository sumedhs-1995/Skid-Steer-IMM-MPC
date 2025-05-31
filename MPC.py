
import cvxpy as opt
import numpy as np


np.seterr(divide="ignore", invalid="ignore")


class MPC:
    def __init__(
        self,
        T                   : float,
        DT                  : float,
        state_multipler     : np.ndarray,
        control_multiplier  : np.ndarray,
        state_cost          : np.ndarray,
        final_state_cost    : np.ndarray,
        input_cost          : np.ndarray,
        input_rate_cost     : np.ndarray,
        initial_state       : np.ndarray
    ):
        """

        Args:
            T                   : Control Horizon (s)
            DT                  : Time Step (s)
            state_multiplier    : A Matrix in x_{k+1} = A.x_{k} + B.u_{k}
            control_multiplier  : B Matrix in x_{k+1} = A.x_{k} + B.u_{k}
            state_cost          : Q Matrix in x_{k}^T.Q.x_{k}
            final_state_cost    : Q_f Matrix in x(T)^T.Q_f.x(T)
            input_cost          : R Matrix in u_{k}^T.R.u_{k}
            input_rate_cost     : P Matrix in Δu_{k}^T.P.Δu_{k} 
        """
        self.nx = 3 # number of state vars
        self.nu = 2 # umber of input/control vars

        if len(state_cost) != self.nx:
            raise ValueError(f"State Error cost matrix should be of size {self.nx}")
        if len(final_state_cost) != self.nx:
            raise ValueError(f"End State Error cost matrix should be of size {self.nx}")
        if len(input_cost) != self.nu:
            raise ValueError(f"Control Effort cost matrix should be of size {self.nu}")
        if len(input_rate_cost) != self.nu:
            raise ValueError(
                f"Control Effort Difference cost matrix should be of size {self.nu}"
            )

        self.dt                 = DT
        self.control_horizon    = int(T / DT)
        self.A                  = state_multipler
        self.B                  = control_multiplier
        self.Q                  = state_cost
        self.Qf                 = final_state_cost
        self.R                  = input_cost
        self.P                  = input_cost

    
    def step(
        self,
        initial_state   : np.ndarray,
        target          : np.ndarray,
        prev_cmd        : np.ndarray,
        verbose         : bool = False,
    ):
        """

        Args:
            initial_state (array-like): current estimate of [x, y, heading]
            target (ndarray): state space reference, in the same frame as the provided current state
            prev_cmd (array-like): previous [acceleration, steer]. note this is used in bounds and has to be realistic.
            verbose (bool):

        Returns:
            x (array-like): predicted optimal state trajectory of size nx * K+1
            u (array-like): predicted optimal control sequence of size nu * K

        """
        assert len(initial_state) == self.nx
        assert len(prev_cmd) == self.nu
        assert target.shape == (self.nx, self.control_horizon)

        # Create variables needed for setting up cvxpy problem
        x = opt.Variable((self.nx, self.control_horizon + 1), name="states")
        u = opt.Variable((self.nu, self.control_horizon), name="actions")
        cost = 0
        constr = [] 

        # Tracking error cost
        for k in range(self.control_horizon):
            cost += opt.quad_form(x[:, k + 1] - target[:, k], self.Q)

        # Final point tracking cost
        cost += opt.quad_form(x[:, -1] - target[:, -1], self.Qf)

        # Actuation magnitude cost
        for k in range(self.control_horizon):
            cost += opt.quad_form(u[:, k], self.R)

        # Actuation rate of change cost
        for k in range(1, self.control_horizon):
            cost += opt.quad_form(u[:, k] - u[:, k - 1], self.P)

        # Kinematics Constrains
        for k in range(self.control_horizon):
            constr += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]

        # initial state
        constr += [x[:, 0] == initial_state]

        # actuation bounds
        constr += [opt.abs(u[:, 0]) <= self.vehicle.max_v]
        constr += [opt.abs(u[:, 1]) <= self.vehicle.max_omega]

        # Actuation rate of change bounds
        constr += [opt.abs(u[0, 0] - prev_cmd[0]) / self.dt <= self.vehicle.max_v]
        constr += [opt.abs(u[1, 0] - prev_cmd[1]) / self.dt <= self.vehicle.max_omega]
        for k in range(1, self.control_horizon):
            constr += [
                opt.abs(u[0, k] - u[0, k - 1]) / self.dt <= self.vehicle.max_v
            ]
            constr += [
                opt.abs(u[1, k] - u[1, k - 1]) / self.dt <= self.vehicle.max_omega
            ]

        prob = opt.Problem(opt.Minimize(cost), constr)
        prob.solve(solver=opt.OSQP, warm_start=True, verbose=False)
        return np.array(x.value), np.array(u.value)
