// Frontend Helper Functions for RAG Chatbot Integration
// Complete implementation with error handling and best practices

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
    // Change this to your deployed backend URL in production
    BACKEND_URL: 'http://localhost:8000',

    // Timeouts and limits
    REQUEST_TIMEOUT: 30000, // 30 seconds
    MAX_MESSAGE_LENGTH: 1000,
    RETRY_ATTEMPTS: 2,

    // UI Configuration
    AUTO_SCROLL: true,
    TYPING_INDICATOR: true,
    TYPING_SPEED: 50, // ms per character (for typing effect)
};

// =============================================================================
// CORE CHAT FUNCTIONALITY
// =============================================================================

/**
 * Main function to handle user messages
 * @param {string} userMessage - The message from the user
 * @param {Object} options - Additional options
 */
async function handleUserMessage(userMessage, options = {}) {
    const trimmedMessage = userMessage.trim();

    // Validate input
    if (!trimmedMessage) {
        displayError('Please enter a message');
        return;
    }

    if (trimmedMessage.length > CONFIG.MAX_MESSAGE_LENGTH) {
        displayError(`Message too long (max ${CONFIG.MAX_MESSAGE_LENGTH} characters)`);
        return;
    }

    // Display user message
    appendMessage('user', trimmedMessage);

    // Show loading indicator
    const loadingId = showLoadingIndicator();

    try {
        // Make API request with retry logic
        const response = await fetchWithRetry('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: trimmedMessage })
        });

        // Remove loading indicator
        removeLoadingIndicator(loadingId);

        // Display assistant response
        if (CONFIG.TYPING_INDICATOR && options.typing !== false) {
            await typeMessage('assistant', response.reply);
        } else {
            appendMessage('assistant', response.reply);
        }

    } catch (error) {
        console.error('Chat error:', error);
        removeLoadingIndicator(loadingId);

        // Display user-friendly error
        const errorMessage = getErrorMessage(error);
        appendMessage('assistant', errorMessage, { isError: true });
    }
}

/**
 * Fetch with retry logic and timeout
 */
async function fetchWithRetry(endpoint, options = {}, attempt = 1) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT);

    try {
        const response = await fetch(`${CONFIG.BACKEND_URL}${endpoint}`, {
            ...options,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        // Handle HTTP errors
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        return await response.json();

    } catch (error) {
        clearTimeout(timeoutId);

        // Retry logic for network errors
        if (attempt < CONFIG.RETRY_ATTEMPTS && isRetryableError(error)) {
            console.log(`Retry attempt ${attempt + 1}...`);
            await sleep(1000 * attempt); // Exponential backoff
            return fetchWithRetry(endpoint, options, attempt + 1);
        }

        throw error;
    }
}

/**
 * Check if error is retryable
 */
function isRetryableError(error) {
    return error.name === 'AbortError' ||
        error.message.includes('network') ||
        error.message.includes('Failed to fetch');
}

/**
 * Convert technical errors to user-friendly messages
 */
function getErrorMessage(error) {
    if (error.name === 'AbortError') {
        return "Request timed out. Please check your connection and try again.";
    }

    if (error.message.includes('Failed to fetch')) {
        return "Couldn't connect to the server. Please check if the backend is running.";
    }

    if (error.message.includes('500')) {
        return "Server error. Please try again in a moment.";
    }

    if (error.message.includes('429')) {
        return "Too many requests. Please wait a moment and try again.";
    }

    return "Sorry, something went wrong. Please try again.";
}

// =============================================================================
// UI MANIPULATION FUNCTIONS
// =============================================================================

/**
 * Append a message to the chat container
 * @param {string} role - 'user' or 'assistant'
 * @param {string} text - Message text
 * @param {Object} options - Additional options (isError, etc.)
 */
function appendMessage(role, text, options = {}) {
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) {
        console.error('Chat container not found');
        return;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    if (options.isError) {
        messageDiv.classList.add('error');
    }

    // Support for markdown-like formatting
    messageDiv.innerHTML = formatMessage(text);

    // Add timestamp if needed
    if (options.timestamp) {
        const timeSpan = document.createElement('span');
        timeSpan.className = 'timestamp';
        timeSpan.textContent = new Date().toLocaleTimeString();
        messageDiv.appendChild(timeSpan);
    }

    chatContainer.appendChild(messageDiv);

    if (CONFIG.AUTO_SCROLL) {
        scrollToBottom(chatContainer);
    }

    return messageDiv;
}

/**
 * Type message with animation effect
 */
async function typeMessage(role, text) {
    const chatContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    chatContainer.appendChild(messageDiv);

    let currentText = '';
    for (const char of text) {
        currentText += char;
        messageDiv.textContent = currentText;

        if (CONFIG.AUTO_SCROLL) {
            scrollToBottom(chatContainer);
        }

        await sleep(CONFIG.TYPING_SPEED);
    }

    // Apply formatting after typing is complete
    messageDiv.innerHTML = formatMessage(text);
}

/**
 * Show loading indicator (three dots animation)
 * @returns {string} - ID of the loading indicator
 */
function showLoadingIndicator() {
    const loadingId = `loading-${Date.now()}`;
    const chatContainer = document.getElementById('chat-messages');

    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message assistant loading';
    loadingDiv.innerHTML = `
    <span class="dot"></span>
    <span class="dot"></span>
    <span class="dot"></span>
  `;

    chatContainer.appendChild(loadingDiv);

    if (CONFIG.AUTO_SCROLL) {
        scrollToBottom(chatContainer);
    }

    return loadingId;
}

/**
 * Remove loading indicator
 */
function removeLoadingIndicator(loadingId) {
    const loadingDiv = document.getElementById(loadingId);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

/**
 * Replace the last message (useful for updating loading states)
 */
function replaceLastMessage(role, text) {
    const chatContainer = document.getElementById('chat-messages');
    const lastMessage = chatContainer.lastElementChild;

    if (lastMessage) {
        lastMessage.className = `message ${role}`;
        lastMessage.innerHTML = formatMessage(text);

        if (CONFIG.AUTO_SCROLL) {
            scrollToBottom(chatContainer);
        }
    }
}

/**
 * Clear all messages from chat
 */
function clearChat() {
    const chatContainer = document.getElementById('chat-messages');
    if (chatContainer) {
        chatContainer.innerHTML = '';
    }
}

/**
 * Display error notification
 */
function displayError(message) {
    // You can customize this to use your own notification system
    console.error(message);

    // Simple alert fallback
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);

    setTimeout(() => errorDiv.remove(), 3000);
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Smooth scroll to bottom of container
 */
function scrollToBottom(container) {
    container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
    });
}

/**
 * Format message text (basic markdown-like support)
 */
function formatMessage(text) {
    // Escape HTML first
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Add basic formatting
    formatted = formatted
        // Bold: **text**
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic: *text*
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Code: `code`
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // Line breaks
        .replace(/\n/g, '<br>');

    return formatted;
}

/**
 * Sleep utility
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// INITIALIZATION
// =============================================================================

/**
 * Initialize the chat interface
 */
function initializeChat() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const clearButton = document.getElementById('clear-button');

    // Send message on button click
    if (sendButton) {
        sendButton.addEventListener('click', () => {
            if (chatInput) {
                handleUserMessage(chatInput.value);
                chatInput.value = '';
            }
        });
    }

    // Send message on Enter key
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleUserMessage(chatInput.value);
                chatInput.value = '';
            }
        });
    }

    // Clear chat button
    if (clearButton) {
        clearButton.addEventListener('click', clearChat);
    }

    // Add welcome message
    appendMessage('assistant', 'Hi! I\'m Mithil\'s AI assistant. Ask me about his projects, skills, or experience!');
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeChat);
} else {
    initializeChat();
}

// =============================================================================
// EXAMPLE CSS (Place in your stylesheet)
// =============================================================================

/*
.message {
  margin: 10px 0;
  padding: 12px 16px;
  border-radius: 8px;
  max-width: 80%;
  word-wrap: break-word;
}

.message.user {
  background: #007bff;
  color: white;
  margin-left: auto;
  text-align: right;
}

.message.assistant {
  background: #f1f1f1;
  color: #333;
  margin-right: auto;
}

.message.error {
  background: #ff4444;
  color: white;
}

.message.loading {
  padding: 20px;
}

.message.loading .dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin: 0 2px;
  background: #666;
  border-radius: 50%;
  animation: loading 1.4s infinite ease-in-out both;
}

.message.loading .dot:nth-child(1) {
  animation-delay: -0.32s;
}

.message.loading .dot:nth-child(2) {
  animation-delay: -0.16s;
}

@keyframes loading {
  0%, 80%, 100% {
    transform: scale(0);
  }
  40% {
    transform: scale(1);
  }
}

.error-notification {
  position: fixed;
  top: 20px;
  right: 20px;
  background: #ff4444;
  color: white;
  padding: 12px 20px;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  z-index: 1000;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

#chat-messages {
  height: 500px;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
}

#chat-input {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: none;
}

#send-button {
  padding: 12px 24px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

#send-button:hover {
  background: #0056b3;
}
*/
