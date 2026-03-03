// Intersection Observer for scroll spy (updating sidebar links)
document.addEventListener('DOMContentLoaded', () => {
    const sections = document.querySelectorAll('.doc-section');
    const navLinks = document.querySelectorAll('.nav-link');

    const observerOptions = {
        root: null,
        rootMargin: '0px 0px -40% 0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');

                // Remove active class from all
                navLinks.forEach(link => {
                    link.classList.remove('active');
                });

                // Add active class to current
                const activeLink = document.querySelector(`.nav-link[href="#${id}"]`);
                if (activeLink) {
                    activeLink.classList.add('active');
                }
            }
        });
    }, observerOptions);

    sections.forEach(section => {
        observer.observe(section);
    });

    // Smooth scrolling for sidebar links
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetExt = document.querySelector(targetId);
            if (targetExt) {
                window.scrollTo({
                    top: targetExt.offsetTop - 50,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Stagger animation delays for sections on load
    sections.forEach((section, index) => {
        section.style.animationDelay = `${0.1 * index}s`;
    });

    // Abstract Canvas Background
    initCanvas();
});

// Minimalist background particle lines
function initCanvas() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    let width, height;
    let nodes = [];

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    class Node {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.3;
            this.vy = (Math.random() - 0.5) * 0.3;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, 1.5, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
            ctx.fill();
        }
    }

    function initNodes() {
        nodes = [];
        const nodeCount = Math.floor((width * height) / 40000); // Responsive density
        for (let i = 0; i < nodeCount; i++) {
            nodes.push(new Node());
        }
    }

    function connectNodes() {
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    // Fade stroke opacity based on distance
                    const opacity = (1 - dist / 150) * 0.05;
                    ctx.strokeStyle = `rgba(255, 255, 255, ${opacity})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);

        nodes.forEach(node => {
            node.update();
            node.draw();
        });

        connectNodes();
        requestAnimationFrame(animate);
    }

    window.addEventListener('resize', () => {
        resize();
        initNodes();
    });

    resize();
    initNodes();
    animate();
}
