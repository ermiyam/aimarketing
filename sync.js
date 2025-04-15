const chokidar = require('chokidar');
const { exec } = require('child_process');
const path = require('path');

class GitSync {
    constructor() {
        this.setupWatcher();
        console.log('🔄 Git Sync Started - Watching for changes...');
    }

    setupWatcher() {
        // Watch all files except .git, node_modules, etc.
        const watcher = chokidar.watch('.', {
            ignored: [
                /(^|[\/\\])\../,
                'node_modules',
                '.git',
                'package-lock.json'
            ],
            persistent: true
        });

        // Handle file events
        watcher
            .on('change', path => this.handleChange(path))
            .on('add', path => this.handleChange(path))
            .on('unlink', path => this.handleChange(path));
    }

    handleChange(filePath) {
        console.log(`📝 Change detected in: ${filePath}`);
        
        // Series of git commands
        const commands = [
            'git add .',
            `git commit -m "Auto-sync: Changes in ${filePath}"`,
            'git pull',  // Pull any changes from other computers
            'git push'   // Push our changes
        ];

        // Execute commands in sequence
        this.executeCommands(commands);
    }

    executeCommands(commands) {
        const command = commands.join(' && ');
        exec(command, (error, stdout, stderr) => {
            if (error) {
                console.error('❌ Sync error:', error);
                return;
            }
            console.log('✅ Changes synchronized successfully');
            if (stdout) console.log(stdout);
        });
    }
}

// Start the sync
new GitSync(); 