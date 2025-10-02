export MAMBA_ROOT_PREFIX="/home/svu/e0588224/Lyapunov/miniconda3"
__mamba_setup="$("/home/svu/e0588224/Lyapunov/miniconda3/bin/mamba" shell hook --shell posix 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="/home/svu/e0588224/Lyapunov/miniconda3/bin/mamba"  # Fallback on help from mamba activate
fi
unset __mamba_setup
