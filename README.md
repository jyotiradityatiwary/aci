# Affective Context Injection

## Running

Clone repository and change directory
```sh
git clone https://github.com/jyotiradityatiwary/aci.git
cd aci
```

Initialize virtual environment:
```sh
python3.11 -m venv .venv
```

Enter virtual environment:
```sh
.venv/bin/activate # bash
```
OR
```sh
.venv/Scripts/activate.ps1 # powershell
```

Install python dependencies
```sh
pip install -r requirements-torch.txt
pip install -r requirements.txt
```

Navigate to `src` subdirectory
```sh
cd src
```

Run Streamlit app
```sh
streamlit run app.py
```
