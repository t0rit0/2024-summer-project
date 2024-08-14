this project is based on the codes provided by the paper below:

@article{andriushchenko2024refusal,
      title={Does Refusal Training in LLMs Generalize to the Past Tense?}, 
      author={Andriushchenko, Maksym and Flammarion, Nicolas},
      journal={arXiv preprint arXiv:2407.11969},
      year={2024}
}

first:
pip install openai
pip install python-dotenv
pip install anthropic
pip install flash-attn



args：
--test: -1: test Over-refusal，  -2：test overrefusal of prompt engineering ， -5：generate lat training set
--n_requests_start: start index
--n_requests_start: end index

## Run experiments
Simply run `main.py`! :-) Examples:
```bash
python main.py --target_model=gpt-3.5-turbo --n_requests=100 --n_restarts=20
python main.py --target_model=gpt-4o-2024-05-13 --n_requests=100 --n_restarts=20 
python main.py --target_model=gpt-4o-mini --n_requests=100 --n_restarts=20 
python main.py --target_model=claude-3-5-sonnet-20240620 --n_requests=100 --n_restarts=20 
python main.py --target_model=phi3 --n_requests=100 --n_restarts=20  
python main.py --target_model=gemma2-9b --n_requests=100 --n_restarts=20 
python main.py --target_model=llama3-8b --n_requests=100 --n_restarts=20 
python main.py --target_model=r2d2 --n_requests=100 --n_restarts=20  
````



@article{andriushchenko2024refusal,
      title={Does Refusal Training in LLMs Generalize to the Past Tense?}, 
      author={Andriushchenko, Maksym and Flammarion, Nicolas},
      journal={arXiv preprint arXiv:2407.11969},
      year={2024}
}


