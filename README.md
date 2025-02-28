Comparação de acerto dos modelos
![image](https://github.com/user-attachments/assets/225ec16c-3ab3-4421-a170-515e8306f837)

Idades por mortes
![image](https://github.com/user-attachments/assets/d0db15af-1461-4893-b3dd-add44a9c6c6f)

Classe de embarcação por morte
![image](https://github.com/user-attachments/assets/4820890c-9e27-4640-b2dd-7210c5198b6d)
 


# Titanic - Kaggle Competition

Este repositório contém a solução para a competição [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) no Kaggle.

## 📌 Descrição do Projeto
O objetivo desta competição é prever quais passageiros sobreviveram ao naufrágio do Titanic, utilizando aprendizado de máquina. Diferentes modelos foram testados, e a melhor solução encontrada foi baseada no **XGBoost**.

## 📂 Estrutura do Repositório
- `contest.py` - Testes exploratórios com diferentes modelos (Regressão Logística, Random Forest, XGBoost).
- - `titanicNEW.py` - Testes exploratórios com dados refinados.
- `submission.py` - Código final usado para gerar o arquivo de submissão para o Kaggle.
- `submission.csv` - Previsões finais geradas pelo modelo.
- `train.csv` e `test.csv` - Dados originais do Titanic.
- `models/` - Pasta opcional para salvar modelos treinados.

## 🛠 Tecnologias Utilizadas
- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- XGBoost

## 🚀 Como Executar
1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/titanic-kaggle.git
   cd titanic-kaggle
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute a submissão final:
   ```bash
   python submission.py
   ```
4. O arquivo `submission.csv` será gerado e pode ser enviado ao Kaggle.

---

# Titanic - Kaggle Competition (English)

This repository contains a solution for the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition on Kaggle.

## 📌 Project Description
The goal of this competition is to predict which passengers survived the Titanic disaster using machine learning. Several models were tested, and the best-performing solution was based on **XGBoost**.

## 📂 Repository Structure
- `contest.py` - Exploratory tests with different models (Logistic Regression, Random Forest, XGBoost).
- `submission.py` - Final code used to generate the submission file for Kaggle.
- `submission.csv` - Final predictions generated by the model.
- `train.csv` and `test.csv` - Original Titanic dataset.
- `models/` - Optional folder to store trained models.

## 🛠 Technologies Used
- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- XGBoost

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/titanic-kaggle.git
   cd titanic-kaggle
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the final submission:
   ```bash
   python submission.py
   ```
4. The `submission.csv` file will be generated and can be submitted to Kaggle.




