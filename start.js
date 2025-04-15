const { spawn } = require('child_process');
const nodeId = process.env.NODE_ID || 'primary';

function startServer() {
    const server = spawn('node', ['src/server.js'], {
        env: { ...process.env, NODE_ID: nodeId },
        stdio: 'inherit'
    });

    server.on('close', (code) => {
        if (code === 1) {
            console.log('Server crashed, restarting...');
            startServer();
        }
    });
}

startServer(); 