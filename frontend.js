import React, { useState } from 'react';
import './LyricsGenerator.css';

const LyricsGenerator = () => {
  const [generatedLyrics, setGeneratedLyrics] = useState('');
  const [seedText, setSeedText] = useState('I\'m feeling');

  const generateLyrics = async () => {
    const response = await fetch('/generate_lyrics', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ seed_text: seedText }),
    });
    const data = await response.json();
    setGeneratedLyrics(data.generated_lyrics);
  };
  

    return (

        <div className="lyrics-generator">
            <h1>AI Lyrics Generator</h1>
            <textarea
                value={seedText}
                onChange={(e) => setSeedText(e.target.value)}
                placeholder="Enter seed text..."
            />
            <button onClick={generatedLyrics}>Generate Lyrics</button>
            <div className="generated-lyrics">

                <h2>Generated Lyrics:</h2>
                <p>{generatedLyrics}</p>
    
            </div>
        </div>
    );
};

export default LyricsGenerator;