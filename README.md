# Jamboree Admission Predictor

> An end-to-end machine learning application that predicts a student's chance of admission to a graduate program from academic and profile-related inputs. The project demonstrates a production-minded workflow: reproducible training, experiment tracking, a REST API, containerization, automated tests, and continuous integration.

[![CI](https://github.com/Ayush-Singh2309/jamboree-admission-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayush-Singh2309/jamboree-admission-predictor/actions/workflows/ci.yml)

**Live Demo:** [https://jamboree-admission-predictor.streamlit.app](https://jamboree-admission-predictor.streamlit.app)  

**Live API:** [https://jamboree-admission-predictor-rfjw.onrender.com](https://jamboree-admission-predictor-rfjw.onrender.com)


## Project Overview

Jamboree Admission Predictor estimates the probability that an applicant will receive a graduate-school admission offer. It packages the full machine learning lifecycle into a maintainable application—from data preparation and model evaluation to an API that can serve predictions.

The project is designed as a portfolio-quality, production-oriented example rather than only a notebook-based analysis.

## Business Problem

Graduate-admission applicants often want a realistic, data-informed estimate of their admission prospects before building their shortlist. This application uses historical applicant information to predict **Chance of Admit**, helping users understand how profile attributes such as GRE score, TOEFL score, university rating, SOP/LOR strength, CGPA, and research experience relate to admission outcomes.

> Predictions are informational estimates, not admissions decisions or guarantees.

## Features

- Reproducible data loading, preprocessing, training, and evaluation pipeline
- Model experimentation with tracked parameters, metrics, and artifacts
- Persisted trained model for repeatable inference
- FastAPI-based REST API with interactive Swagger documentation
- Input validation through Pydantic schemas
- Automated tests with `pytest`
- Docker support for consistent local and deployment environments
- CI workflow with GitHub Actions

## Project Architecture

![architecture](images/Architecture.png)

## Folder Structure

```text
jamboree-admission-predictor/
├── app/                    # FastAPI application and API schemas
├── frontend/               # Streamlit application and dashboard
├── src/                    # Training, preprocessing, and utility modules
├── tests/                  # Unit and API tests
├── notebooks/              # Exploratory data analysis notebooks
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned or transformed data
├── artifacts/              # Model, metrics, plots, and predictions
├── .github/workflows/      # GitHub Actions CI configuration
├── Dockerfile
├── requirements.txt
├── README.md
└── LICENSE
```

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data & ML | pandas, NumPy, scikit-learn |
| Experiment tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Testing | pytest |
| Containerization | Docker |
| CI | GitHub Actions |
| Frontend | Streamlit |
| Deployment | Render |

## Dataset

The model is trained on the **Jamboree Admission** dataset, which contains applicant-level academic and profile features along with the target variable, `Chance of Admit`.

Typical input features include:

- GRE Score
- TOEFL Score
- University Rating
- Statement of Purpose (SOP) strength
- Letter of Recommendation (LOR) strength
- Undergraduate CGPA
- Research experience

Dataset source: [Kaggle — Graduate Admissions](https://www.kaggle.com/code/imprime/graduate-admissions-dataset)


## Experimentation

Multiple regression models can be evaluated to select the most reliable approach for the problem. Typical candidates include Linear Regression, Ridge Regression, Lasso Regression.

Models should be compared using held-out test data and regression metrics such as R², RMSE, and MAE. The selected model is then serialized and used by the inference API.


| Model | Test R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.81884 | 0.06087 | 0.04272 |
| Ridge Regression | 0.81854 | 0.06092 | 0.04284 |
| Lasso Regression | 0.81914 | 0.06082 | 0.04255 |


## Training Pipeline

The training workflow follows these stages:

1. Load the raw dataset.
2. Validate columns and clean missing or inconsistent values.
3. Split the data into training and test sets.
4. Fit the preprocessing and model pipeline.
5. Evaluate performance on unseen test data.
6. Save the trained model and evaluation artifacts.
7. Log parameters, metrics, and artifacts to MLflow.

TRAINING PIPELINE DIAGRAM
```text
              Training Pipeline

        raw DATA (Train, Validation, Test)
                      │
                      ▼
             Validation/Cleaning/Splitting
                      │
                      ▼
            Preprocessing/Model-Selection
                      │
                      ▼
                 train.py
                      │
     ┌────────────────┼─────────────────┐
     ▼                ▼                 ▼
Evaluation       Artifacts         MLflow
     │                │                 │
     ▼                ▼                 ▼
Metrics        Model + Plots      Experiment Logs
     │                │
     └────────────┬───┘
                  ▼
             model.pkl
```

## MLflow Tracking

MLflow records each experiment run, including model parameters, evaluation metrics, and generated artifacts. This makes it easier to compare experiments and reproduce the selected model.

Start the MLflow UI locally with:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser.

Experiments:
![MLflow Experiments](images/experiments.png)

Runs:
![MLflow Runs](images/runs.png)

Artifacts:
![MLflow Artifacts](images/artifacts.png)

## FastAPI API

The trained model is exposed through a FastAPI service. Once running, interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Swagger UI:
[Link](https://jamboree-admission-predictor-rfjw.onrender.com/docs)
![Swagger](images/swagger.png)


## Docker

Docker packages the application and its dependencies into a portable image, ensuring the API behaves consistently across environments.

Docker Image:
![Docker Image](images/image.png)

Docker Container:
![Docker Container](images/container.png)

## CI/CD

GitHub Actions runs automated checks on pushes and pull requests. A typical workflow installs dependencies, runs tests, builds the Docker image and can optionally deploy the service.

![GitHub Actions](images/CI.png)

## Installation

### Prerequisites

- Python 3.12 or later
- `pip`
- Docker (optional, for containerized usage)

### Clone and set up

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt
```

## Local Usage

### Train the model

```bash
python -m src.train
```

Training produces the saved model and evaluation outputs in `artifacts/`.

### Serve the API

```bash
uvicorn app.api:app --reload
```

Open the Swagger UI at `http://127.0.0.1:8000/docs`.


### Run tests

```bash
python -m pytest
```

## Docker Commands

Build the image:

```bash
docker build -t jamboree-admission .
```

Run the container:

```bash
docker run -d -p 5000:9000 --name jamboree-admission-predictor jamboree-admission
```

Visit `http://127.0.0.1:9000/docs` after the container starts.


## API Examples

### Health check

```bash
curl http://127.0.0.1:9000/health
```

### Prediction request

```bash
curl -X POST "http://127.0.0.1:9000/predict" \
  -H "Content-Type: application/json" \
  -d '{
  "GRE_Score": 320,
  "TOEFL_Score": 110,
  "University_Rating": 4,
  "SOP": 4.0,
  "LOR": 4.0,
  "CGPA": 9.0,
  "Research": 1
}'
```

Example response:

```json
{
  "chance_of_admit": 0.8098
}
```

## Results

| Metric | Value |
| --- | ---: |
| Test R² | 0.81914 |
| Test RMSE | 0.06082 |
| Test MAE | 0.04255 |

![Actual vs Predicted Plot](artifacts/plots/actual_vs_predicted_plot.png)
![QQ Plot](artifacts/plots/qq_plot.png)
![Residual Distribution Plot](artifacts/plots/residual_distribution.png)
![Residual Plot](artifacts/plots/residual_plot.png)

## Future Improvements

- Add data and model validation checks with tools such as Great Expectations or Evidently.
- Add hyperparameter optimization and model selection automation.
- Introduce model and data versioning with DVC.
- Add monitoring for prediction drift and API performance after deployment.

## Acknowledgements

- Jamboree Education / the original dataset provider for the admissions dataset.
- The open-source communities behind scikit-learn, FastAPI, MLflow, Docker, and pytest.

## License

This project is licensed under the [MIT License](LICENSE).
