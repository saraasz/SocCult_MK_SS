echo "Installing requirements"
pip install -r "requirements.txt"
echo "Running simulations"
nohup python3 -u simulate.py "configs/machine$1.cfg" &> "machine$1.out" &
echo "Watching output (you can quit this the process will continue in the background)"
watch cat "machine$1.out"
