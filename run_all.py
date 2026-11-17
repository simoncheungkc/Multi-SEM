"""End-to-end runner for the reported offline displays.

Simulation, the covariance-adapted Boston splits, the S&P 100 panel, and the
joint LM diagnostics. The S&P script stays on the frozen snapshot: live prices
require ``python run_sp100.py --download`` and are not started from here.
"""

from run_boston import main as boston_main
from run_joint_lm import main as joint_lm_main
from run_simulation import main as sim_main
from run_sp100 import main as sp_main


if __name__ == "__main__":
    sim_main()
    boston_main(adaptive=True)
    sp_main([])
    joint_lm_main()
