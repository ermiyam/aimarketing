const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔧 Starting dependency fix...');

// 1. Install fs-extra specifically
try {
    console.log('Installing fs-extra...');
    execSync('npm install fs-extra --save', { stdio: 'inherit' });
    console.log('✅ fs-extra installed');
} catch (error) {
    console.error('Failed first attempt, trying alternative installation...');
}

// 2. Update package.json to ensure fs-extra is listed
const packagePath = path.join(process.cwd(), 'package.json');
let packageJson = require(packagePath);

packageJson.dependencies = {
    ...packageJson.dependencies,
    'fs-extra': '^10.0.0'
};

fs.writeFileSync(packagePath, JSON.stringify(packageJson, null, 2));
console.log('✅ Updated package.json');

// 3. Clean install
try {
    console.log('Cleaning node_modules...');
    execSync('rm -rf node_modules package-lock.json', { stdio: 'inherit' });
    
    console.log('Reinstalling all dependencies...');
    execSync('npm install', { stdio: 'inherit' });
    
    console.log('✅ Dependencies reinstalled');
} catch (error) {
    console.error('Error during reinstall:', error);
}

console.log('🎉 Fix complete! Try running your app now.'); 