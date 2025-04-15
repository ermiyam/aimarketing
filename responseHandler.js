const axios = require('axios');
const natural = require('natural');
const tokenizer = new natural.WordTokenizer();
const LearningManager = require('./services/LearningManager');

class ResponseHandler {
  constructor() {
    this.knowledgeBase = new Map();
    this.classifier = new natural.BayesClassifier();
    this.initializeClassifier();
  }

  initializeClassifier() {
    // Add training data for intent classification
    this.classifier.addDocument('hi hello hey', 'greeting');
    this.classifier.addDocument('what is how to explain', 'question');
    this.classifier.addDocument('lost missing file', 'file_loss');
    this.classifier.addDocument('marketing campaign strategy', 'marketing_query');
    this.classifier.train();
  }

  async handleResponse(message) {
    try {
      // First, check existing knowledge
      const response = await this.checkKnowledgeBase(message);
      if (response) return response;

      // If not found, learn from the internet
      const learnedInfo = await LearningManager.learnFromQuery(message);
      
      return {
        type: 'learned_response',
        content: learnedInfo.summary,
        sources: learnedInfo.sources.map(s => s.url)
      };
    } catch (error) {
      return {
        type: 'error',
        content: 'I encountered an error while learning. Please try again.'
      };
    }
  }
  
  async handleMarketingQuery(message) {
    try {
      // Example: Fetch relevant data from Wikipedia
      const searchTerm = encodeURIComponent(message);
      const response = await axios.get(
        `https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro=1&explaintext=1&titles=${searchTerm}`
      );
      
      // Process and return the response
      return {
        type: 'marketing_response',
        content: response.data.query.pages[Object.keys(response.data.query.pages)[0]].extract
      };
    } catch (error) {
      return {
        type: 'error',
        content: 'Sorry, I couldn\'t fetch that information. Please try again.'
      };
    }
  }

  async checkKnowledgeBase(message) {
    const keywords = LearningManager.extractKeywords(message);
    const key = keywords.join('_');
    
    if (this.knowledgeBase.has(key)) {
      const info = this.knowledgeBase.get(key);
      info.accessCount++;
      info.lastAccessed = Date.now();
      return {
        type: 'cached_response',
        content: info.summary
      };
    }
    
    return null;
  }
}

module.exports = new ResponseHandler(); 