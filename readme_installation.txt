
1. Install requirements.txt
2. Add this project to your python path, e.g. in your bashrc
3. Use pyenv for easy switching between python/virtualenv versions

For Car Racing 
-> pip3 install Box2D
-> pip3 install gym[box2d]
-> if you do not have swig -> sudo apt install swig

2 virtual envs need to be installed (and switching with python3 versions)
    1. For RL expert generation using old libraries
        pyenv virtualenv 3.7.9 baselines
        cd rl/
        pyenv local baselines
        pip3 install stable-baselines[mpi]
        pip3 install tensorflow==1.15

    2. Using current libraries for imitation learning
