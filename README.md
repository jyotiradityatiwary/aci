# Affective Context Injection

## Running

1. Clone repository and change directory
```sh
git clone https://github.com/jyotiradityatiwary/aci.git
cd aci
```

2. Initialize virtual environment:
```sh
python3.11 -m venv .venv
```

3. Enter virtual environment:
```sh
.venv/bin/activate # bash
```
OR
```sh
.venv/Scripts/activate.ps1 # powershell
```

4. Install python dependencies
```sh
pip install -r requirements-torch.txt
pip install -r requirements.txt
```

5. Navigate to `src` subdirectory
```sh
cd src
```

6. Copy and rename `config.py.example` to  `config.py`
```sh
cp config.py.example config.py
```

7. Enter you api keys in `config.py`

8. Run Streamlit app
```sh
streamlit run app.py
```
