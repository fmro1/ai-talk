

# ROCm Setup for Conversational AI
echo "Setting up ROCm support for Conversational AI..."

# Activate virtual environment
source venv/bin/activate

# Check ROCm installation
echo "Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo "✓ ROCm detected"
    rocm-smi --showproductname
else
    echo "⚠ ROCm not detected. Please install ROCm first:"
    echo "  https://docs.amd.com/en/latest/deploy/linux/installer/install.html"
fi

# Check PyTorch ROCm support
echo "Checking PyTorch ROCm support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No CUDA/ROCm support detected')
"

# Install ROCm-compatible PyTorch if needed
echo ""
read -p "Install ROCm-compatible PyTorch? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch for ROCm..."
    
    # Uninstall existing PyTorch
    pip uninstall -y torch torchaudio torchvision
    
    # Install ROCm version (adjust ROCm version as needed)
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
    
    echo "PyTorch ROCm installation complete!"
fi

# Test the setup
echo ""
echo "Testing ROCm setup with Whisper..."
python -c "
try:
    from faster_whisper import WhisperModel
    import torch
    
    if torch.cuda.is_available():
        print('Testing tiny model on GPU...')
        model = WhisperModel('tiny.en', device='cuda', compute_type='float16')
        print('✓ ROCm Whisper test successful!')
    else:
        print('⚠ GPU not available, testing CPU...')
        model = WhisperModel('tiny.en', device='cpu', compute_type='int8') 
        print('✓ CPU Whisper test successful!')
        
except Exception as e:
    print(f'⚠ Whisper test failed: {e}')
"

echo ""
echo "ROCm setup complete!"
echo ""
echo "To use ROCm in your config.json, set:"
echo '  "device": "cuda"'
echo '  "compute_type": "float16"'
echo ""
echo "If you encounter issues, try:"
echo '  "device": "cpu"'
echo '  "compute_type": "int8"'