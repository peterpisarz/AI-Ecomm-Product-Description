// JavaScript for AI Product Description Generator

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add loading state to submit buttons
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                // Add spinner and change text
                const originalContent = submitButton.innerHTML;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Generating...';
                submitButton.disabled = true;
                
                // Add loading class to animate
                submitButton.classList.add('loading');
                
                // Restore if form validation fails
                setTimeout(() => {
                    if (!form.checkValidity()) {
                        submitButton.innerHTML = originalContent;
                        submitButton.disabled = false;
                        submitButton.classList.remove('loading');
                    }
                }, 100);
            }
        });
    });

    // Make alert dismissible
    const alertCloseButtons = document.querySelectorAll('.alert .btn-close');
    alertCloseButtons.forEach(button => {
        button.addEventListener('click', function() {
            const alert = this.closest('.alert');
            if (alert) {
                alert.style.opacity = '0';
                setTimeout(() => {
                    alert.style.display = 'none';
                }, 150);
            }
        });
    });

    // Add copy to clipboard functionality
    setupCopyButtons();
    
    // Add toast notifications
    setupToasts();
    
    // Add form validation styles
    setupFormValidation();
});

// Function to set up copy buttons
function setupCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            // Prevent default only if it's a link
            if (button.tagName === 'A') {
                event.preventDefault();
            }
            
            // Get text to copy (either from data attribute or from target element)
            let textToCopy;
            if (button.dataset.target) {
                const targetElement = document.getElementById(button.dataset.target);
                if (targetElement) {
                    textToCopy = targetElement.innerText;
                }
            } else if (button.dataset.copy) {
                textToCopy = button.dataset.copy;
            }
            
            // If text was found, copy it
            if (textToCopy) {
                navigator.clipboard.writeText(textToCopy)
                    .then(() => {
                        // Change button text temporarily
                        const originalText = button.innerHTML;
                        button.innerHTML = '<i class="fas fa-check me-1"></i> Copied!';
                        
                        // Restore original text after a short delay
                        setTimeout(() => {
                            button.innerHTML = originalText;
                        }, 2000);
                    })
                    .catch(err => {
                        console.error('Could not copy text: ', err);
                    });
            }
        });
    });
}

// Function to set up toast notifications
function setupToasts() {
    // Show toasts when triggered
    const toastTriggers = document.querySelectorAll('[data-bs-toggle="toast"]');
    toastTriggers.forEach(trigger => {
        trigger.addEventListener('click', function() {
            const toastTarget = document.getElementById(trigger.dataset.bsTarget);
            if (toastTarget) {
                const toast = new bootstrap.Toast(toastTarget);
                toast.show();
            }
        });
    });
}

// Function to set up form validation
function setupFormValidation() {
    // Add custom validation styles
    const formsToValidate = document.querySelectorAll('.needs-validation');
    formsToValidate.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
        
        // Add custom validation message for product name (if it exists)
        const productNameInput = form.querySelector('#product_name');
        if (productNameInput) {
            productNameInput.addEventListener('input', function() {
                if (productNameInput.value.trim() === '') {
                    productNameInput.setCustomValidity('Please enter a product name');
                } else {
                    productNameInput.setCustomValidity('');
                }
            });
        }
    });
}

// Function to animate the SEO score progress bars
function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const width = bar.getAttribute('aria-valuenow') + '%';
        bar.style.width = '0%';
        
        // Trigger reflow
        bar.offsetWidth;
        
        // Animate to actual width
        bar.style.transition = 'width 1s ease-in-out';
        bar.style.width = width;
    });
}

// Call animate progress bars if they exist on page load
if (document.querySelectorAll('.progress-bar').length > 0) {
    window.addEventListener('load', animateProgressBars);
} 