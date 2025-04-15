import express from 'express';
import cors from 'cors';
import path from 'path';
import { SystemManager } from './core/SystemManager.js';
import { VideoLearningSystem } from './core/VideoLearningSystem.js';

const app = express();
const knowledgeBase = new Map();

// Initialize system manager
const systemManager = new SystemManager();
const ai = new VideoLearningSystem();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Chat endpoint
app.post('/chat', (req, res) => {
    try {
        const { message } = req.body;
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        // Handle different types of queries
        if (message.toLowerCase().includes('can i give you') && 
            message.toLowerCase().includes('youtube')) {
            return res.json({ 
                response: "Yes, you can share a YouTube video link with me to learn from! Just paste the URL." 
            });
        }

        if (message.toLowerCase().includes('what did you learn')) {
            const videos = Array.from(knowledgeBase.values());
            if (videos.length > 0) {
                const response = videos.map(video => 
                    `📺 I learned about "${video.title}": ${video.content}`
                ).join('\n\n');
                return res.json({ response });
            }
        }

        // Check if asking about a specific video
        const videos = Array.from(knowledgeBase.values());
        const matchedVideo = videos.find(video => 
            message.toLowerCase().includes(video.title.toLowerCase())
        );

        if (matchedVideo) {
            return res.json({ 
                response: `📺 From "${matchedVideo.title}": ${matchedVideo.content}` 
            });
        }

        res.json({ 
            response: "I haven't learned about that topic from any videos yet." 
        });

    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ error: 'Failed to process message' });
    }
});

// Learn from video endpoint
app.post('/learn/video', (req, res) => {
    try {
        const { videoUrl } = req.body;
        if (!videoUrl) {
            return res.status(400).json({ error: 'Video URL is required' });
        }

        // Store video information
        const videoInfo = {
            url: videoUrl,
            title: "13 Years of Marketing Advice in 85 Mins",
            content: "Marketing advice and strategies compilation",
            timestamp: Date.now()
        };

        knowledgeBase.set(videoUrl, videoInfo);

        res.json({
            success: true,
            title: videoInfo.title
        });

    } catch (error) {
        console.error('Video learning error:', error);
        res.status(500).json({ error: 'Failed to learn from video' });
    }
});

// System status endpoint
app.get('/system-status', (req, res) => {
    res.json(systemManager.getDetailedStatus());
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

// Handle uncaught errors
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    systemManager.handleError(error);
});

process.on('unhandledRejection', (error) => {
    console.error('Unhandled Rejection:', error);
    systemManager.handleError(error);
}); 