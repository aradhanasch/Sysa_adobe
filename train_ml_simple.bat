@echo off
echo Training ML model...
docker run --rm -v "D:/hackathon_adobe/app/input:/app/input" -v "D:/hackathon_adobe/app/output:/app/output" pdf-extractor python app/train_ml.py
echo.
echo ML training completed!
pause 