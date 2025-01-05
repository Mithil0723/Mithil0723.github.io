// Typing Animation
const roles = ['Data Science Student', 'Machine Learning Enthusiast', 'Problem Solver'];
let roleIndex = 0;
let charIndex = 0;
let isDeleting = false;
let typedTextElement = document.getElementById('typed-text');

function typeText() {
    const currentRole = roles[roleIndex];
    
    if (!isDeleting && charIndex <= currentRole.length) {
        typedTextElement.textContent = currentRole.substring(0, charIndex);
        charIndex++;
    } else if (isDeleting && charIndex >= 0) {
        typedTextElement.textContent = currentRole.substring(0, charIndex);
        charIndex--;
    }

    if (charIndex === currentRole.length + 1) {
        setTimeout(() => {
            isDeleting = true;
        }, 2000);
    }

    if (charIndex === -1) {
        isDeleting = false;
        roleIndex = (roleIndex + 1) % roles.length;
        charIndex = 0;
    }

    const typingSpeed = isDeleting ? 100 : 200;
    setTimeout(typeText, typingSpeed);
}

// Start the typing animation
typeText();

// Header Toggle
let MenuBtn = document.getElementById("MenuBtn");
let header = document.querySelector("header");

MenuBtn.addEventListener('click', function() {
    document.body.classList.toggle('move-nav-active');
    this.classList.toggle('fa-xmark');
});

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
            // Close mobile menu if open
            if (window.innerWidth <= 1050) {
                document.body.classList.remove('move-nav-active');
                MenuBtn.classList.remove('fa-xmark');
            }
        }
    });
});

const form = document.querySelector('.contact-form');

form.addEventListener('submit', function (event) {
  const name = form.querySelector('input[name="name"]').value.trim();
  const email = form.querySelector('input[name="email"]').value.trim();
  const message = form.querySelector('textarea[name="message"]').value.trim();

  console.log("Name:", name);
  console.log("Email:", email);
  console.log("Message:", message);

  if (!name || !email || !message) {
    event.preventDefault();
    alert('Please fill out all fields before submitting.');
  }
});

// Add these to your main.js file

// Skills section animation
document.addEventListener('DOMContentLoaded', function() {
    const skillCategories = document.querySelectorAll('.skill-category');
    
    // Add animation when skills come into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    skillCategories.forEach(category => {
        category.style.opacity = 0;
        category.style.transform = 'translateY(20px)';
        category.style.transition = 'all 0.5s ease';
        observer.observe(category);
    });

    // Handle PDF loading errors
    const resumeIframe = document.querySelector('.resume-preview iframe');
    if (resumeIframe) {
        resumeIframe.onerror = function() {
            this.style.display = 'none';
            const errorMessage = document.createElement('div');
            errorMessage.className = 'resume-error';
            errorMessage.innerHTML = `
                <p>Unable to load PDF preview. Please use the buttons below to download or view the resume.</p>
            `;
            this.parentNode.appendChild(errorMessage);
        };
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const projectImages = document.querySelectorAll('.project-img');
    
    projectImages.forEach(img => {
        img.onerror = function() {
            this.src = '/api/placeholder/400/320'; // Fallback to placeholder
            this.alt = 'Project Preview Unavailable';
        };
    });
});

// Add this to your main.js file
document.addEventListener('DOMContentLoaded', function() {
    // Handle project detail links
    document.querySelectorAll('.details-btn').forEach(button => {
        button.addEventListener('click', async function(e) {
            e.preventDefault();
            const detailsUrl = this.getAttribute('href');
            
            try {
                const response = await fetch(detailsUrl);
                if (response.ok) {
                    const content = await response.text();
                    showProjectModal(content);
                } else {
                    console.error('Failed to load project details');
                }
            } catch (error) {
                console.error('Error loading project details:', error);
            }
        });
    });
});

function showProjectModal(content) {
    // Create modal container if it doesn't exist
    let modal = document.querySelector('.project-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.className = 'project-modal';
        document.body.appendChild(modal);
    }

    // Set modal content and show it
    modal.innerHTML = `
        <div class="modal-content">
            <button class="modal-close">&times;</button>
            <div class="modal-body">${content}</div>
        </div>
    `;

    // Add modal styles dynamically
    const style = document.createElement('style');
    style.textContent = `
        .project-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            overflow-y: auto;
        }
        .modal-content {
            position: relative;
            background-color: var(--sidebgColor);
            margin: 50px auto;
            padding: 20px;
            width: 90%;
            max-width: 800px;
            border-radius: 10px;
            color: white;
        }
        .modal-close {
            position: absolute;
            right: 20px;
            top: 10px;
            font-size: 30px;
            background: none;
            border: none;
            color: white;
            cursor: pointer;
        }
        .modal-body {
            margin-top: 20px;
        }
    `;
    document.head.appendChild(style);

    // Show modal
    modal.style.display = 'block';

    // Add close button functionality
    modal.querySelector('.modal-close').addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Close on outside click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// Add this to your main.js file

// Add this to your main.js file

document.addEventListener('DOMContentLoaded', function() {
    // Replace all "Learn More" links with buttons and add click handlers
    document.querySelectorAll('.details-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const projectCard = this.closest('.project-card');
            const detailsSection = projectCard.querySelector('.project-details');
            
            // Toggle the details section
            if (detailsSection.style.maxHeight) {
                detailsSection.style.maxHeight = null;
                this.innerHTML = '<i class="fa-solid fa-circle-info"></i> Learn More';
                detailsSection.classList.remove('active');
            } else {
                detailsSection.style.maxHeight = detailsSection.scrollHeight + "px";
                this.innerHTML = '<i class="fa-solid fa-chevron-up"></i> Show Less';
                detailsSection.classList.add('active');
            }
        });
    });
});

document.addEventListener('DOMContentLoaded', () => {
    const toggleBtns = document.querySelectorAll('.details-toggle-btn');

    toggleBtns.forEach(button => {
        button.addEventListener('click', () => {
            const projectDetails = button.closest('.project-details');
            projectDetails.classList.toggle('expanded');

            if (projectDetails.classList.contains('expanded')) {
                button.textContent = 'Show Less';
            } else {
                button.textContent = 'Show More';
            }
        });
    });
});
