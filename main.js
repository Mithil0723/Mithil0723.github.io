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
