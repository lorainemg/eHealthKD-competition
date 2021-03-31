train_baseline: 
	python3 scripts/baseline.py --ref 2021/ref/training --eval 2021/eval/training --submit 2021/submissions/baseline/training/run1

evaluate_baseline:
	 python3 scripts/score.py --gold 2021/eval/training --submit 2021/submissions/baseline/training 

running_baseline:
	python3 scripts/baseline.py --ref 2021/ref/training --eval 2021/eval/develop --submit 2021/submissions/baseline/develop/run1

evaluating_baseline:
	python3 scripts/score.py --gold 2021/eval/develop --submit 2021/submissions/baseline/develop
