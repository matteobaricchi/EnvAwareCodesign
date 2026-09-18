# Environmental-aware wind farm layout optimization and control co-design
Optimization framework for environmental-aware wind farm layout optimization, wind farm flow control, and wind farm co-design. 

This repository provides the figures, data, and code associated with the following research article:

>Baricchio, M., Kainz, S., van Wingerden, J.W., and Bottasso, C. L.: Environmental-aware wind farm layout optimization and control co-design, Wind Energy Science (under preparation), 2026.

The repository contains:

1. An example Python script for performing multi-objective layout design optimization and layout-control co-design optimization. The provided code contains the optimization framework, while the objective function used in this study cannot be made publicly available due to licensed emission factors. For questions regarding the latter, please contact Samuel Kainz at the Technical University of Munich (samuel.kainz@tum.de).
2. The Python script used to generate the data and plots for two illustrative examples with 2 and 3 turbines, respectively. The resulting datasets are also provided, allowing the script to be used for plotting purposes only (see the corresponding flag in the input parameters).
3. The results from the multi-objective layout design and layout-control co-design optimizations, as well as the optimized yaw angles.
4. A script for analyzing the multi-objective optimization results, which was used to create the corresponding figures in the paper.
5. A list of package versions used to run the above-mentioned scripts.
6. PDF versions of all figures used in the paper, pickled Matplotlib plots for most figures, and the underlying data for figures for which neither the pickled plots nor the plotting code can be provided.

We hope you find the code and data useful. May the wind be strong, the wakes be weak, and the optimization converge.

Matteo Baricchio and Samuel Kainz
