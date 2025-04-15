const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔧 Starting emergency fix...');

// 1. Define all required dependencies
const dependencies = {
    "express": "^4.17.1",
    "cors": "^2.8.5",
    "mongoose": "^6.0.0",
    "natural": "^5.1.13",
    "fs-extra": "^10.0.0",
    "axios": "^0.21.1",
    "nodemon": "^2.0.7"
};

// 2. Create fresh package.json
const packageJson = {
    "name": "ai-marketing-app",
    "version": "1.0.0",
    "main": "index.js",
    "scripts": {
        "start": "node index.js",
        "dev": "nodemon index.js"
    },
    "dependencies": dependencies
};

// 3. Write package.json
fs.writeFileSync('package.json', JSON.stringify(packageJson, null, 2));
console.log('✅ Created fresh package.json');

// 4. Clean install
try {
    console.log('Cleaning previous installation...');
    execSync('rm -rf node_modules package-lock.json');
    
    console.log('Installing dependencies...');
    execSync('npm install', { stdio: 'inherit' });
    
    console.log('✅ Dependencies installed successfully');
} catch (error) {
    console.error('Installation error:', error);
}

console.log('🎉 Fix complete! Run "npm start" to start the server.'); 