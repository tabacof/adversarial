# Host computer
# Go to .pem directory
chmod 400 *.pem
ssh -i tabacof.pem ubuntu@dns

# Install Torch7
curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh
source ~/.bashrc

# Clone project
git clone https://github.com/tabacof/adversarial.git

# Install BLAS
sudo apt-get install libatlas-base-dev

# Make LBFGSB lib
cd adversarial/lbfgsb
make lib

# Download OverFeat weights
cd ../overfeat
sh install.sh
th run.lua

# Test project
th adversarial.lua

# Install iTorch

# Install CUDA

# Install GPU Torch libraries

# Install gfx.js
sudo apt-get install node npm
sudo apt-get install xdg-utils
sudo apt-get install links links2 lynx
sudo apt-get install libgraphicsmagick1-dev
sudo apt-get install graphicsmagick
luarocks install https://raw.github.com/clementfarabet/gfx.js/master/gfx.js-scm-0.rockspec

