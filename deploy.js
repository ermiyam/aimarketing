const pm2 = require('pm2');

pm2.connect((err) => {
    if (err) {
        console.error(err);
        process.exit(2);
    }

    pm2.start({
        name: 'ai-sync-server',
        script: 'src/server-manager.js',
        watch: true,
        env: {
            NODE_ENV: 'production',
            PORT: 3001
        }
    }, (err) => {
        if (err) {
            console.error('Error starting server:', err);
            return;
        }

        console.log('24/7 server deployed successfully');
        pm2.disconnect();
    });
}); 