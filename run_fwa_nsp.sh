#!/bin/bash

# Ativa o ambiente virtual no Windows (modo compatível com Git Bash ou WSL)
source env/Scripts/activate

# Executa o FWA para o Nurse Scheduling Problem com parâmetros definidos
python main_nsp.py \
  --xml_path data/ORTEC01.xml \
  --save_file results/fwa_nsp_run_06 \
  --fwa_n 10 \
  --fwa_m 100 \
  --fwa_a 0.03 \
  --fwa_b 0.9 \
  --fwa_a_hat 60 \
  --fwa_m_hat 10 \
  --fwa_max_iter 1000 \
  --fwa_select_mode distance

# Pausar o terminal após a execução
read -p "Pressione [Enter] para sair..."
