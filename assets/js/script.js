/**
 * script.js — Redesigned portfolio
 * - No magnetic cursor
 * - No 3D card tilt
 * - Simple translateY hover via CSS
 * - IntersectionObserver reveal animations
 * - Hamburger menu toggle
 * - Chat integration with frontend-helpers.js
 */

document.addEventListener('DOMContentLoaded', function () {

    // =========================================================================
    // HAMBURGER MENU
    // =========================================================================

    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('nav-links');

    if (hamburger && navLinks) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navLinks.classList.toggle('active');
        });

        // Also toggle on Enter/Space for a11y
        hamburger.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                hamburger.classList.toggle('active');
                navLinks.classList.toggle('active');
            }
        });

        // Close menu when a nav link is clicked
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navLinks.classList.remove('active');
            });
        });
    }

    // =========================================================================
    // REVEAL ON SCROLL (IntersectionObserver)
    // =========================================================================

    const revealElements = document.querySelectorAll('.reveal');
    if (revealElements.length > 0) {
        // Respect prefers-reduced-motion
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        if (prefersReducedMotion) {
            // If reduced motion, just show everything immediately
            revealElements.forEach(el => {
                el.classList.add('revealed');
            });
        } else {
            const revealObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('revealed');
                    }
                });
            }, {
                threshold: 0.1,
                rootMargin: "0px 0px -50px 0px"
            });

            revealElements.forEach(el => {
                revealObserver.observe(el);
            });
        }
    }

    // =========================================================================
    // CHAT INTEGRATION (hooks into frontend-helpers.js)
    // =========================================================================

    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');

    if (chatInput && sendBtn && chatMessages) {
        async function sendMessage() {
            const message = chatInput.value.trim();
            if (!message) return;

            // Display user message
            const userDiv = document.createElement('div');
            userDiv.className = 'q';
            userDiv.textContent = '> ' + message;
            chatMessages.appendChild(userDiv);
            chatInput.value = '';
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Show loading
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'a loading-dots';
            loadingDiv.innerHTML = '<span></span><span></span><span></span>';
            chatMessages.appendChild(loadingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                // Use CONFIG from frontend-helpers.js if available
                const backendUrl = (typeof CONFIG !== 'undefined' && CONFIG.BACKEND_URL)
                    ? CONFIG.BACKEND_URL
                    : 'https://rag-chatbot-6boa.onrender.com';

                const controller = new AbortController();
                const timeout = setTimeout(() => controller.abort(), 30000);

                const sessionId = sessionStorage.getItem('chat_session_id');
                const bodyPayload = { message: message };
                if (sessionId) {
                    bodyPayload.session_id = sessionId;
                }

                const response = await fetch(`${backendUrl}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(bodyPayload),
                    signal: controller.signal,
                });

                clearTimeout(timeout);

                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }

                const data = await response.json();
                
                if (data.session_id) {
                    sessionStorage.setItem('chat_session_id', data.session_id);
                }
                
                loadingDiv.remove();

                const answerDiv = document.createElement('div');
                answerDiv.className = 'a';
                answerDiv.textContent = data.reply || data.response || data.answer || 'No response received.';
                chatMessages.appendChild(answerDiv);
            } catch (err) {
                loadingDiv.remove();
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error';
                errorDiv.textContent = err.name === 'AbortError'
                    ? 'Request timed out. Please try again.'
                    : 'Could not reach the AI agent. Try again later.';
                chatMessages.appendChild(errorDiv);
            }

            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // =========================================================================
    // SMOOTH SCROLL for nav links (already handled by CSS scroll-behavior,
    // but this ensures it works with the # links properly)
    // =========================================================================

    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});