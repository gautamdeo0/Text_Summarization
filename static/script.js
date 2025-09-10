document.addEventListener('DOMContentLoaded', function() {
    const processBtn = document.getElementById('processBtn');
    const translationResult = document.getElementById('translationResult');
    const originalSummary = document.getElementById('originalSummary');
    const translatedSummary = document.getElementById('translatedSummary');

    function setLoading(button, isLoading) {
        if (isLoading) {
            button.classList.add('loading');
            button.disabled = true;
            button.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...`;
        } else {
            button.classList.remove('loading');
            button.disabled = false;
            button.innerHTML = `<i class="bi bi-gear-fill me-2"></i>Process Text`;
        }
    }

    function showResult(element, text, isError = false) {
        element.style.animation = 'none';
        element.offsetHeight; // Trigger reflow
        element.style.animation = 'fadeIn 0.5s ease-in-out';
        element.textContent = text;
        if (isError) {
            element.style.color = '#dc3545';
            element.style.borderColor = '#dc3545';
        } else {
            element.style.color = 'inherit';
            element.style.borderColor = '#e9ecef';
        }
    }

    function clearResults() {
        translationResult.textContent = '';
        originalSummary.textContent = '';
        translatedSummary.textContent = '';
    }

    // Text processing functionality
    processBtn.addEventListener('click', async function() {
        const text = document.getElementById('inputText').value;
        const direction = document.getElementById('translationDirection').value;
        
        if (!text) {
            alert('Please enter text to process');
            return;
        }

        try {
            setLoading(processBtn, true);
            clearResults();

            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    direction: direction
                })
            });

            const data = await response.json();
            
            if (data.error) {
                showResult(translationResult, data.error, true);
            } else {
                showResult(translationResult, data.translation);
                showResult(originalSummary, data.original_summary);
                showResult(translatedSummary, data.translated_summary);
            }
        } catch (error) {
            console.error('Error:', error);
            showResult(translationResult, 'An error occurred while processing. Please try again.', true);
            clearResults();
        } finally {
            setLoading(processBtn, false);
        }
    });

    // Add keydown event listener for textarea
    document.getElementById('inputText').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            processBtn.click();
        }
    });
}); 