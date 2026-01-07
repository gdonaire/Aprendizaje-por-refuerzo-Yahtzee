# Aprendizaje-por-refuerzo-Yahtzee


# Instalar entorno virtual

Python -m venv /home/gdonaire/UOC/05.629_TFG_Inteligencia_artificial_Aula1

# Acivar entorno virtual

Source venv/bin/actívate

# Clonar repositorio 

git clone https://github.com/gdonaire/Aprendizaje-por-refuerzo-Yahtzee.git

# Instalar dependencias
Pip install -r requirements.txt
Pip install -e .

# Entrenar algoritmos
python run_variants_yahtzee.py --model a2c --variants a2c_baseline,a2c_highhorizon_lowvar,a2c_longrollout,a2c_fastexploit --num_seeds 3 --run_name a2c_variants_no_repair --n_envs 8 --eval_episodes 50

python run_variants_yahtzee.py --model ppo --variants ppo_baseline,ppo_highhorizon_lowvar,ppo_conservative_clip_kl,ppo_big_net --num_seeds 3 --run_name ppo_variants_no_repair --n_envs 8 --eval_episodes 50

python run_variants_yahtzee.py --model mppo --variants mppo_baseline,mppo_highhorizon_lowvar,mppo_conservative_clip_kl,mppo_big_net --num_seeds 3 --run_name mppo_variants --n_envs 8 --eval_episodes 50

python run_variants_yahtzee.py --model dqn --variants dqn_baseline,dqn_batch_big,dqn_target_slow_grad_more,dqn_nstep5_epsfloor010 --num_seeds 3 --run_name dqn_variants_no_repair --n_envs 8 --eval_episodes 50

python run_variants_yahtzee.py --model qrdqn --variants qrdqn_baseline,qrdqn_batch_big,qrdqn_target_slow_grad_more,qrdqn_nstep5_epsfloor010 --num_seeds 3 --run_name qrdqn_variants_no_repair --n_envs 8 --eval_episodes 50

# Comparativa final
python compare_algorithms_eval_only.py \
		--a2c_report runs/a2c_variants/a2c_variants_report.md \
		--ppo_report runs/ppo_variants/ppo_variants_report.md \
	 	--dqn_report runs/dqn_variants/dqn_variants_report.md \
		--mppo_report runs/mppo_variants/mppo_variants_report.md \
		--qrdqn_report runs/qrdqn_variants/qrdqn_variants_report.md \
         	--global_run_name final_compare_evalonly \
         	--episodes 200 --max_steps 1000 --seed 42
