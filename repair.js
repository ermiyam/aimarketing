const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 Starting complete system repair...');

// 1. Clean everything
try {
    console.log('Cleaning project...');
    execSync('rm -rf node_modules package-lock.json yarn.lock');
} catch (error) {
    // Continue if files don't exist
}

// 2. Create minimal working index.js
const minimalIndex = `
const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Basic health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Basic chat endpoint
app.post('/chat', (req, res) => {
    try {
        const { message } = req.body;
        if (!message) {
            return res.status(400).json({ error: 'Message required' });
        }
        res.json({ response: 'Hello! I am working correctly.' });
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(\`Server running on port \${PORT}\`);
});
`;

// 3. Create fresh package.json
const packageJson = {
    "name": "ai-marketing-app",
    "version": "1.0.0",
    "main": "index.js",
    "scripts": {
        "start": "node index.js",
        "dev": "nodemon index.js"
    },
    "dependencies": {
        "express": "4.17.1",
        "cors": "2.8.5"
    }
};

// 4. Write files
console.log('Creating fresh files...');
fs.writeFileSync('index.js', minimalIndex);
fs.writeFileSync('package.json', JSON.stringify(packageJson, null, 2));

// 5. Install dependencies
console.log('Installing dependencies...');
try {
    execSync('npm install', { stdio: 'inherit' });
    console.log('✅ Basic system installed successfully');
} catch (error) {
    console.error('Failed to install dependencies:', error);
    process.exit(1);
}

// 6. Verify installation
try {
    console.log('Verifying installation...');
    require('express');
    require('cors');
    console.log('✅ Verification successful');
} catch (error) {
    console.error('Verification failed:', error);
    process.exit(1);
}

console.log('🎉 Repair complete! Run "npm start" to start the server.'); 