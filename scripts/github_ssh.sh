#!/bin/bash

echo "Generating a new SSH key for GitHub..."

mkdir -p ~/.ssh
chmod 700 ~/.ssh


ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "a7medalghali777@gmail.com" -N ""


echo "Writing SSH config..."
cat > ~/.ssh/config <<EOF
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF


chmod 600 ~/.ssh/config

# Show the public key
echo ""
echo "📋 Copy the following PUBLIC key and add it to GitHub → Settings → SSH and GPG Keys:"
echo "────────────────────────────────────────────"
cat ~/.ssh/id_ed25519.pub
echo "────────────────────────────────────────────"
echo ""
echo "🔐 Done! After adding the key to GitHub, test with:"
echo "ssh -T git@github.com"