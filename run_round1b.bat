@echo off
echo Running Round 1B: Persona-Driven Analysis...
docker run --rm -v "D:/hackathon_adobe/app/input:/app/input" -v "D:/hackathon_adobe/app/output:/app/output" pdf-trainer python app/round1b/main_round1b.py
echo.
echo Round 1B completed!
pause 