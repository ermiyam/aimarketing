const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const axios = require('axios');
const cheerio = require('cheerio');
const natural = require('natural');
const { Worker } = require('worker_threads');

class AutonomousAI {
    constructor() {
        this.name = "Gen";
        this.version = "12.0";
        this.learningWorkers = new Map();
        this.memorySystem = new AdvancedMemorySystem();
        this.webCrawler = new WebLearningSystem();
        this.knowledgeGraph = new KnowledgeGraph();
        this.learningMetrics = new LearningMetrics();
        
        // Initialize all systems
        this.initialize();
    }

    async initialize() {
        console.log('Initializing Autonomous Learning Systems...');
        await this.initializeSubsystems();
        this.startAutonomousLearning();
    }

    async initializeSubsystems() {
        // Initialize memory system
        await this.memorySystem.initialize();
        
        // Start knowledge graph
        await this.knowledgeGraph.initialize();
        
        // Initialize learning workers
        this.initializeLearningWorkers();
        
        // Start web crawler
        await this.webCrawler.initialize();
    }

    initializeLearningWorkers() {
        const workerTypes = [
            'semantic', 'pattern', 'neural',
            'evolutionary', 'creative', 'analytical'
        ];

        workerTypes.forEach(type => {
            const worker = new Worker('./workers/learningWorker.js', {
                workerData: { type }
            });
            
            worker.on('message', this.handleWorkerMessage.bind(this));
            worker.on('error', this.handleWorkerError.bind(this));
            
            this.learningWorkers.set(type, worker);
        });
    }

    startAutonomousLearning() {
        // Start continuous web learning
        this.webCrawler.startContinuousLearning();
        
        // Start knowledge processing
        this.knowledgeGraph.startProcessing();
        
        // Start memory optimization
        this.memorySystem.startOptimization();
        
        // Monitor and adjust learning parameters
        this.startLearningMonitor();
    }

    async handleWorkerMessage(message) {
        if (message.type === 'learning_result') {
            await this.processLearningResult(message.data);
        } else if (message.type === 'new_pattern') {
            await this.knowledgeGraph.addPattern(message.data);
        }
    }

    handleWorkerError(error) {
        console.error('Learning worker error:', error);
        // Implement recovery strategy
        this.recoverWorker(error.workerId);
    }

    startLearningMonitor() {
        setInterval(() => {
            this.optimizeLearningParameters();
            this.checkLearningProgress();
            this.adjustLearningStrategies();
        }, 5000); // Check every 5 seconds
    }
}

class WebLearningSystem {
    constructor() {
        this.learningQueue = new Set();
        this.processedUrls = new Set();
        this.learningTopics = new Map();
        this.currentDepth = 0;
        this.maxDepth = 3;
    }

    async initialize() {
        // Initial learning seeds
        const initialTopics = [
            'artificial intelligence',
            'machine learning',
            'data science',
            'neural networks',
            'cognitive science'
        ];

        for (const topic of initialTopics) {
            await this.addTopic(topic);
        }
    }

    async startContinuousLearning() {
        while (true) {
            try {
                const url = await this.getNextUrl();
                if (url) {
                    await this.learnFromUrl(url);
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
            } catch (error) {
                console.error('Learning error:', error);
                continue;
            }
        }
    }

    async learnFromUrl(url) {
        try {
            const { data } = await axios.get(url);
            const $ = cheerio.load(data);
            
            // Extract text content
            const textContent = $('p, h1, h2, h3, h4, h5, h6')
                .map((_, el) => $(el).text())
                .get()
                .join(' ');

            // Process content
            await this.processContent(textContent, url);
            
            // Extract and queue new URLs
            this.queueNewUrls($);
            
            // Mark URL as processed
            this.processedUrls.add(url);
            
        } catch (error) {
            console.error(`Error learning from ${url}:`, error);
        }
    }

    async processContent(content, url) {
        // Tokenize and analyze content
        const tokens = new natural.WordTokenizer().tokenize(content);
        const tfidf = new natural.TfIdf();
        tfidf.addDocument(tokens);

        // Extract key concepts
        const concepts = this.extractConcepts(tfidf);
        
        // Update knowledge graph
        await this.updateKnowledge(concepts, url);
    }

    extractConcepts(tfidf) {
        const concepts = new Map();
        tfidf.listTerms(0).forEach(item => {
            concepts.set(item.term, item.tfidf);
        });
        return concepts;
    }

    async updateKnowledge(concepts, url) {
        // Send to main memory system
        await this.memorySystem.storeConcepts(concepts, url);
        
        // Update learning topics
        concepts.forEach((weight, concept) => {
            if (weight > 0.5) { // Significant concepts
                this.learningTopics.set(concept, {
                    weight,
                    lastSeen: Date.now(),
                    sources: new Set([url])
                });
            }
        });
    }
}

class AdvancedMemorySystem {
    constructor() {
        this.shortTermMemory = new Map();
        this.longTermMemory = new Map();
        this.workingMemory = new Map();
        this.memoryIndex = new Map();
    }

    async initialize() {
        await this.loadMemoryState();
        this.startMemoryManagement();
    }

    startMemoryManagement() {
        // Periodic memory consolidation
        setInterval(() => {
            this.consolidateMemory();
        }, 60000); // Every minute

        // Memory optimization
        setInterval(() => {
            this.optimizeMemory();
        }, 300000); // Every 5 minutes
    }

    async consolidateMemory() {
        for (const [key, value] of this.shortTermMemory) {
            if (this.shouldConsolidate(value)) {
                await this.moveToLongTerm(key, value);
            }
        }
    }

    shouldConsolidate(memory) {
        return memory.accessCount > 5 || 
               memory.importance > 0.7 ||
               Date.now() - memory.firstSeen > 3600000; // 1 hour
    }

    async moveToLongTerm(key, value) {
        this.longTermMemory.set(key, {
            ...value,
            consolidationDate: Date.now()
        });
        this.shortTermMemory.delete(key);
        await this.saveMemoryState();
    }

    optimizeMemory() {
        // Remove outdated memories
        this.cleanupMemory();
        
        // Reorganize memory index
        this.reindexMemory();
        
        // Compress rarely accessed memories
        this.compressMemory();
    }
}

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static('public'));

const gen = new AutonomousAI();

app.post('/api/think', async (req, res) => {
    try {
        const response = await gen.think(req.body.input);
        res.json(response);
    } catch (error) {
        res.status(500).json({ error: 'Thinking error occurred' });
    }
});

app.get('/api/stats', (req, res) => {
    res.json({
        intelligence: gen.intelligence,
        stats: gen.stats
    });
});

app.listen(port, () => {
    console.log(`SuperAI running at http://localhost:${port}`);
}); 