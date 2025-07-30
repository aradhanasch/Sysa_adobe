@echo off
docker run --rm -v "D:/hackathon_adobe/app/input:/app/input" -v "D:/hackathon_adobe/app/output:/app/output" pdf-extractor
pause 