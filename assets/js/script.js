document.addEventListener('DOMContentLoaded', function () {

    // Typing animation
    const typingText = document.querySelector('.typing-text');
    if (typingText) {
        const texts = JSON.parse(typingText.getAttribute('data-texts'));
        let textIndex = 0;
        let charIndex = 0;

        function type() {
            if (charIndex < texts[textIndex].length) {
                const char = texts[textIndex].charAt(charIndex);
                if (char === ' ') {
                    typingText.appendChild(document.createTextNode(' '));
                } else {
                    const span = document.createElement('span');
                    span.className = 'char-in';
                    span.textContent = char;
                    typingText.appendChild(span);
                }
                charIndex++;
                setTimeout(type, 100);
            } else {
                setTimeout(erase, 2000);
            }
        }

        function erase() {
            if (charIndex > 0) {
                if (typingText.lastChild) {
                    typingText.removeChild(typingText.lastChild);
                }
                charIndex--;
                setTimeout(erase, 50);
            } else {
                textIndex = (textIndex + 1) % texts.length;
                setTimeout(type, 500);
            }
        }

        typingText.innerHTML = '';
        type();
    }

    // Magnetic Cursor
    const cursor = document.querySelector('.magnetic-cursor');
    const cursorFollower = document.querySelector('.magnetic-cursor-follower');

    // Only run on non-touch devices
    if (cursor && cursorFollower && window.matchMedia('(pointer: fine)').matches) {
        let mouseX = window.innerWidth / 2;
        let mouseY = window.innerHeight / 2;
        let cursorX = mouseX;
        let cursorY = mouseY;
        let followerX = mouseX;
        let followerY = mouseY;

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        const render = () => {
            cursorX += (mouseX - cursorX) * 1;
            cursorY += (mouseY - cursorY) * 1;

            followerX += (mouseX - followerX) * 0.15;
            followerY += (mouseY - followerY) * 0.15;

            cursor.style.transform = `translate3d(${cursorX}px, ${cursorY}px, 0) translate(-50%, -50%)`;
            cursorFollower.style.transform = `translate3d(${followerX}px, ${followerY}px, 0) translate(-50%, -50%)`;

            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);

        // Hover effects
        const interactables = document.querySelectorAll('a, button, .skill-card, .project-card, input, textarea');
        interactables.forEach(el => {
            el.addEventListener('mouseenter', () => {
                cursor.classList.add('hover');
                cursorFollower.classList.add('hover');
            });
            el.addEventListener('mouseleave', () => {
                cursor.classList.remove('hover');
                cursorFollower.classList.remove('hover');
            });
        });
    }

    // Sticky header
    const header = document.querySelector('.header');
    if (header) {
        window.addEventListener('scroll', function () {
            if (window.scrollY > 50) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        });
    }

    // Hamburger menu
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });

        document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        }));
    }

    // Reveal Animations
    const revealElements = document.querySelectorAll('.reveal');
    if (revealElements.length > 0) {
        const revealObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');

                    // Animate skill bars if they exist inside this revealed element
                    const bars = entry.target.querySelectorAll('.skill-bar');
                    bars.forEach(bar => {
                        // slight delay to let the card fade in first
                        setTimeout(() => {
                            bar.style.width = bar.getAttribute('data-level');
                        }, 500);
                    });
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


    // Parallax Depth on Cards
    const tiltCards = document.querySelectorAll('.project-card, .skill-card');
    tiltCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            // Only apply tilt on non-touch devices
            if (!window.matchMedia('(pointer: fine)').matches) return;

            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = ((y - centerY) / centerY) * -10;
            const rotateY = ((x - centerX) / centerX) * 10;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });

        card.addEventListener('mouseleave', () => {
            if (!window.matchMedia('(pointer: fine)').matches) return;
            card.style.transform = `perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)`;
            card.style.transition = 'transform 0.5s ease, box-shadow 0.5s ease';
        });

        card.addEventListener('mouseenter', () => {
            if (!window.matchMedia('(pointer: fine)').matches) return;
            card.style.transition = 'none';
        });
    });

    // Button Ripple Effect
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', function (e) {
            const x = e.clientX - e.target.getBoundingClientRect().left;
            const y = e.clientY - e.target.getBoundingClientRect().top;

            const ripples = document.createElement('span');
            ripples.style.left = x + 'px';
            ripples.style.top = y + 'px';
            ripples.classList.add('ripple');

            this.appendChild(ripples);

            setTimeout(() => {
                ripples.remove();
            }, 600);
        });
    });

    // Nav underline morph
    const navLinks = document.querySelectorAll('.nav-link');
    const navMenuMorph = document.querySelector('.nav-menu');

    if (navMenuMorph && window.innerWidth > 768) {
        const marker = document.createElement('div');
        marker.classList.add('nav-marker');
        navMenuMorph.appendChild(marker);

        function indicator(e) {
            marker.style.left = e.offsetLeft + 'px';
            marker.style.width = e.offsetWidth + 'px';
        }

        navLinks.forEach(link => {
            link.addEventListener('mouseenter', (e) => {
                indicator(e.target);
            });
        });

        navMenuMorph.addEventListener('mouseleave', () => {
            marker.style.width = '0px';
        });
    }
    const filterBtns = document.querySelectorAll('.filter-btn');
    const projectCards = document.querySelectorAll('.project-card');

    if (filterBtns.length > 0 && projectCards.length > 0) {
        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Set active button
                filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const filter = btn.getAttribute('data-filter');

                // Show/hide projects with animation
                projectCards.forEach(card => {
                    if (filter === 'all' || card.getAttribute('data-category') === filter) {
                        card.style.display = 'flex'; // It's a flex container
                        setTimeout(() => {
                            card.style.opacity = '1';
                            card.style.transform = 'scale(1)';
                        }, 10);
                    } else {
                        card.style.opacity = '0';
                        card.style.transform = 'scale(0.8)';
                        setTimeout(() => {
                            if (card.style.opacity === '0') {
                                card.style.display = 'none';
                            }
                        }, 300);
                    }
                });
            });
        });
    }

    /*-----------------------------------*\
     * #CHATBOT logic is now handled by frontend-helpers.js
    \*-----------------------------------*/
});