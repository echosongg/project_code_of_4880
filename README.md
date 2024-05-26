#Network Construction and Pattern Detection

This project consists of two main parts: network construction (`network_contraction`) and pattern detection (`pattern_detection`). Below is a detailed description of each part and its components.

## network_contraction

### File Description

- `net_cons.py`: Responsible for interpolation and selection of year, latitude and longitude to be used.
- `built_net.py`: used to build the entire undirected weighted network. The generated graph file is stored in `graph_Aus_mask_3year_thresh2_same_t_and_r.graphml`.

## pattern_detection

### File Description

- `Louvain.py`: Implements community detection of Louvain method. Results and log information are stored in the `l_result` folder.
- `propagation.py`: implements community detection of propagation algorithm. Results and log information are stored in the `p_result` folder.
- `spectral.py`: implements community detection using the spectral method. Results and log information are stored in the `s_result` folder.

### Result visualization

The results of each community detection method include visual partitioning results to facilitate users to understand and analyze the community structure.

## Instructions

Currently, specific usage instructions have not been added to this `README.md`. It is recommended to consult the comments within each script for more details on how to execute these scripts.

## contribute

Any suggestions or contributions to improve the project are welcome. Please contact us via Issues or Pull Requests.

## License

No permits have been assigned to this project. Please make sure you understand the relevant laws and terms before use.
