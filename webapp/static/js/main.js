document.addEventListener('DOMContentLoaded', function() {
  const analyzeButton = document.getElementById('analyzeButton');
  const inputText = document.getElementById('inputText');
  const sentimentDisplay = document.getElementById('sentiment-text');
  const emojiDisplay = document.getElementById('emoji');
  const scoresDisplay = document.getElementById('confidenceScores');
  const resultBox = document.getElementById('result');
  const aboutBtn = document.getElementById('aboutBtn');
  const aboutModal = document.getElementById('aboutModal');
  const modalClose = document.querySelector('.modal-close');

  // Add to existing DOMContentLoaded function




  // Verify elements are found
  console.log("Elements found:", {
    analyzeButton: !!analyzeButton,
    inputText: !!inputText,
    sentimentDisplay: !!sentimentDisplay,
    emojiDisplay: !!emojiDisplay,
    scoresDisplay: !!scoresDisplay,
    resultBox: !!resultBox
  });

  // Map sentiment labels to emoji and color
  const sentimentMapping = {
    'Positive': { emoji: '😃', color: '#4CAF50' },
    'Neutral': { emoji: '😐', color: '#FFC107' },
    'Negative': { emoji: '😞', color: '#F44336' },
    'Irrelevant': { emoji: '🤷', color: '#9E9E9E' }
  };

  analyzeButton.addEventListener('click', function() {
    const text = inputText.value.trim();
    console.log("Input text:", text);

    if (!text) {
      alert('Please enter some text.');
      return;
    }

    // Show loading state
    analyzeButton.disabled = true;
    analyzeButton.textContent = 'Analyzing...';
    
    // Clear previous results and show loading state
    resultBox.classList.add('hidden');
    sentimentDisplay.textContent = '';
    emojiDisplay.textContent = '';
    scoresDisplay.innerHTML = '';

    fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: text })
    })
    .then(response => {
      console.log("Response status:", response.status);
      return response.json();
    })
    .then(data => {
      console.log("Received data:", data);
      
      if (data.error) {
        throw new Error(data.error);
      }

      const sentiment = data.sentiment;
      console.log("Processing sentiment:", sentiment);
      
      // Update sentiment display and emoji
      sentimentDisplay.textContent = sentiment;
      const mapping = sentimentMapping[sentiment] || { emoji: '', color: '#f6d365' };
      emojiDisplay.textContent = mapping.emoji;
      console.log("Updated display elements");

      // Create score bars
      let scoresHtml = '';
      for (const [label, score] of Object.entries(data.scores)) {
        const percentage = Math.round(score * 100);
        const barColor = (sentimentMapping[label] && sentimentMapping[label].color) || '#f6d365';
        scoresHtml += `
          <div class="score">
            <div class="score-label">${label}: ${percentage}%</div>
            <div class="progress-bar">
              <div class="progress-fill" style="width: ${percentage}%; background: ${barColor};"></div>
            </div>
          </div>
        `;
      }
      scoresDisplay.innerHTML = scoresHtml;
      console.log("Updated scores display");
      
      // Show the result box
      resultBox.classList.remove('hidden');
      console.log("Removed hidden class from result box");
    })
    .catch(error => {
      console.error("Error:", error);
      sentimentDisplay.textContent = 'Error: ' + error.message;
      scoresDisplay.innerHTML = '';
    })
    .finally(() => {
      // Reset button state
      analyzeButton.disabled = false;
      analyzeButton.textContent = 'Analyze';
    });
  });

  // Ensure modal is hidden at start
  if (aboutModal) {
    aboutModal.classList.add('hidden');
  }

  // About button click handler
  aboutBtn.addEventListener('click', () => {
    aboutModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  });

  // Close button click handler
  if (modalClose) {
    modalClose.addEventListener('click', (e) => {
      e.stopPropagation(); // Prevent event bubbling
      closeModal();
    });
  }

  // Close modal when clicking outside
  aboutModal.addEventListener('click', (e) => {
    if (e.target === aboutModal) {
      closeModal();
    }
  });

  // Close modal function
  function closeModal() {
    aboutModal.classList.add('hidden');
    document.body.style.overflow = 'auto';
  }

  // Close modal with Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !aboutModal.classList.contains('hidden')) {
      closeModal();
    }
  });
}); 
//DOMContentLoaded
