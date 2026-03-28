#!/bin/bash
# Start SSH daemon in background
/usr/sbin/sshd

# If PUBLIC_KEY is set (RunPod injects this), add it
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
fi

# Forward all args to vllm serve
exec vllm serve "$@"
