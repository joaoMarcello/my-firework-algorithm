#!/bin/bash

# Ativa o ambiente virtual no Windows (modo compatível com Git Bash ou WSL)
# source env/Scripts/activate
# conda activate fwa_paa

# Executa o FWA para o Nurse Scheduling Problem com parâmetros definidos

  # -
    # --xml_path benchmark/Instance2.xml \
python main_nsp.py \
  --xml_path data/ORTEC01.xml \
  --save_file results_tests_17-07/fwa_nsp_run_01 \
  --fwa_n 20 \
  --fwa_m 30 \
  --fwa_a 0.04 \
  --fwa_b 0.8 \
  --fwa_a_hat 1.5 \
  --fwa_m_hat 5 \
  --fwa_j 0.2 \
  --fwa_j_hat 0.5 \
  --fwa_max_iter 500 \
  --fwa_select_mode roulette

# Pausar o terminal após a execução
read -p "Pressione [Enter] para sair..."
