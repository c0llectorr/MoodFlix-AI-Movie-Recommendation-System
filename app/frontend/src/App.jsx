import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Heart, Frown, Smile, Zap, Meh, AlertCircle, TrendingUp, Moon, Sun } from 'lucide-react';
import MovieCarousel from './components/MovieCarousel';

const EmotionDetector = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [movies, setMovies] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [isDark, setIsDark] = useState(true);
  const textareaRef = useRef(null);

  // API endpoint - change this to your deployed backend URL
  const API_URL = 'https://mahmdshafee-emotion-detection-api.hf.space';

  // Emotion configurations with colors and icons
  const emotionConfig = {
    joy: { color: '#FFD700', gradient: 'from-yellow-400 to-amber-500', icon: Smile, emoji: '😊' },
    love: { color: '#FF69B4', gradient: 'from-pink-400 to-rose-500', icon: Heart, emoji: '❤️' },
    surprise: { color: '#9370DB', gradient: 'from-purple-400 to-violet-500', icon: Zap, emoji: '😲' },
    neutral: { color: '#94A3B8', gradient: 'from-slate-400 to-gray-500', icon: Meh, emoji: '😐' },
    sadness: { color: '#4682B4', gradient: 'from-blue-400 to-cyan-600', icon: Frown, emoji: '😢' },
    anger: { color: '#DC143C', gradient: 'from-red-500 to-orange-600', icon: AlertCircle, emoji: '😠' },
    fear: { color: '#8B4513', gradient: 'from-amber-700 to-orange-800', icon: TrendingUp, emoji: '😨' }
  };

  // Check API health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        const data = await response.json();
        setApiHealth(data);
      }
    } catch (err) {
      console.error('Health check failed:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    if (text.length > 5000) {
      setError('Text too long (max 5000 characters)');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setMovies(null);

    try {
      // Call the new /recommendations endpoint
      const response = await fetch(`${API_URL}/recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setResult({
        emotion: data.emotion,
        confidence: data.confidence,
      });
      setMovies(data.recommendations);
      
    } catch (err) {
      setError(err.message || 'Failed to connect to server');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTextChange = (e) => {
    setText(e.target.value);
    setError(null);
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setMovies(null);
    setError(null);
    textareaRef.current?.focus();
  };

  const tryExample = (exampleText) => {
    setText(exampleText);
    setError(null);
    setResult(null);
    setMovies(null);
  };

  const exampleTexts = [
    "I'm so excited about my vacation next week!",
    "This situation makes me really frustrated and angry.",
    "I miss my family so much, feeling lonely today.",
    "You mean everything to me, I love you.",
    "I'm worried about the exam results tomorrow.",
    "Just another regular day at work.",
    "Wow! I can't believe this just happened!"
  ];

  // Dark theme: Blade Runner 2049 - Neon pink/purple cyberpunk
  // Light theme: Joker 2019 - Neon orange suit inspired
  const bgLight = 'bg-gradient-to-br from-orange-50 via-amber-50 to-red-50';
  const bgDark = 'bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950';
  
  const headerLight = 'bg-gradient-to-r from-orange-500 via-red-500 to-orange-600 border-orange-400';
  const headerDark = 'bg-gradient-to-r from-fuchsia-900 via-purple-900 to-violet-900 border-fuchsia-600';
  
  const cardLight = 'bg-white border-orange-400 shadow-lg';
  const cardDark = 'bg-slate-900/80 border-fuchsia-500 shadow-2xl shadow-fuchsia-500/20';

  const sortedEmotions = result 
    ? Object.entries(result.all_probabilities || {}).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className={`min-h-screen transition-colors duration-300 ${isDark ? bgDark : bgLight}`}>
      {/* Header */}
      <div className={`border-b backdrop-blur-sm shadow-md ${isDark ? headerDark : headerLight}`}>
        <div className="max-w-6xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${isDark ? 'bg-gradient-to-br from-fuchsia-500 to-purple-600' : 'bg-gradient-to-br from-orange-500 to-red-600'}`}>
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className={`text-2xl font-bold ${isDark ? 'text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-400 to-cyan-400' : 'text-transparent bg-clip-text bg-gradient-to-r from-orange-600 to-red-600'}`}>
                  MoodFlix
                </h1>
                <p className={`text-xs ${isDark ? 'text-fuchsia-300' : 'text-orange-700'}`}>Emotion-based Movie Recommendations</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {apiHealth && (
                <div className={`hidden sm:flex items-center space-x-2 px-3 py-1 rounded-full border ${isDark ? 'bg-cyan-950/50 border-cyan-600' : 'bg-orange-50 border-orange-400'}`}>
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className={`text-xs font-medium ${isDark ? 'text-cyan-400' : 'text-orange-700'}`}>
                    API Online
                  </span>
                </div>
              )}
              
              {/* Theme Toggle */}
              <button
                onClick={() => setIsDark(!isDark)}
                className={`p-2 rounded-lg transition-all ${isDark ? 'bg-fuchsia-900/50 hover:bg-fuchsia-800 text-cyan-300 border border-fuchsia-600' : 'bg-orange-300 hover:bg-orange-400 text-gray-900 border border-orange-500'}`}
              >
                {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-6 sm:px-6 lg:px-8 sm:py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Left Column - Input */}
          <div className="lg:col-span-2 space-y-6">
            {/* Input Card */}
            <div className={`rounded-2xl border-2 overflow-hidden ${isDark ? cardDark : cardLight}`}>
              <div className="p-6">
                <h2 className={`text-lg font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  How are you feeling?
                </h2>
                
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="relative">
                    <textarea
                      ref={textareaRef}
                      value={text}
                      onChange={handleTextChange}
                      placeholder="Share your thoughts or feelings... (e.g., 'I'm feeling amazing today!')"
                      className={`w-full h-32 sm:h-40 px-4 py-3 border-2 rounded-xl focus:ring-4 transition-all resize-none ${
                        isDark
                          ? 'bg-slate-800 border-fuchsia-500 focus:border-cyan-400 focus:ring-fuchsia-500/20 text-white placeholder-slate-500'
                          : 'bg-white border-orange-400 focus:border-red-500 focus:ring-orange-200 text-gray-900 placeholder-gray-400'
                      }`}
                      disabled={loading}
                    />
                    <div className={`absolute bottom-3 right-3 text-xs ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
                      {text.length} / 5000
                    </div>
                  </div>

                  {error && (
                    <div className={`flex items-center space-x-2 p-3 border rounded-lg ${isDark ? 'bg-red-950/50 border-red-600' : 'bg-red-50 border-red-300'}`}>
                      <AlertCircle className={`w-4 h-4 flex-shrink-0 ${isDark ? 'text-red-400' : 'text-red-600'}`} />
                      <p className={`text-sm ${isDark ? 'text-red-300' : 'text-red-700'}`}>{error}</p>
                    </div>
                  )}

                  <div className="flex flex-col sm:flex-row gap-3">
                    <button
                      type="submit"
                      disabled={loading || !text.trim()}
                      className={`flex-1 flex items-center justify-center space-x-2 px-6 py-3 rounded-xl font-medium shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none ${
                        isDark
                          ? 'bg-gradient-to-r from-fuchsia-600 to-purple-600 text-white hover:from-fuchsia-700 hover:to-purple-700 border border-fuchsia-500'
                          : 'bg-gradient-to-r from-orange-500 to-red-500 text-white hover:from-orange-600 hover:to-red-600 border border-orange-400'
                      }`}
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          <span>Analyzing & Finding Movies...</span>
                        </>
                      ) : (
                        <>
                          <Send className="w-5 h-5" />
                          <span>Detect & Suggest</span>
                        </>
                      )}
                    </button>
                    
                    <button
                      type="button"
                      onClick={handleClear}
                      disabled={loading}
                      className={`px-6 py-3 rounded-xl font-medium transition-colors disabled:opacity-50 ${
                        isDark
                          ? 'bg-slate-800 text-slate-200 hover:bg-slate-700 border border-fuchsia-500'
                          : 'bg-orange-200 text-gray-800 hover:bg-orange-300 border border-orange-400'
                      }`}
                    >
                      Clear
                    </button>
                  </div>
                </form>
              </div>
            </div>

            {/* Results Card */}
            {result && (
              <div className={`rounded-2xl border-2 overflow-hidden animate-fadeIn ${isDark ? cardDark : cardLight}`}>
                <div className="p-6 space-y-6">
                  {/* Primary Emotion */}
                  <div>
                    <h2 className={`text-lg font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                      Your Emotion
                    </h2>
                    <div className={`relative p-6 rounded-xl bg-gradient-to-r ${emotionConfig[result.emotion].gradient} overflow-hidden`}>
                      <div className="absolute top-0 right-0 text-8xl opacity-10">
                        {emotionConfig[result.emotion].emoji}
                      </div>
                      <div className="relative">
                        <div className="flex items-center space-x-3 mb-2">
                          {React.createElement(emotionConfig[result.emotion].icon, {
                            className: "w-8 h-8 text-white"
                          })}
                          <h3 className="text-3xl font-bold text-white capitalize">
                            {result.emotion}
                          </h3>
                        </div>
                        <p className="text-white/90 text-lg">
                          Confidence: {(result.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Movies Carousels */}
            {movies && movies.length > 0 && (
              <div className="space-y-4 animate-fadeIn">
                <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  Movies For You
                </h2>
                {movies.map((genreMovies, idx) => (
                  <MovieCarousel
                    key={idx}
                    genre={genreMovies.genre}
                    movies={genreMovies.movies}
                    isDark={isDark}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Right Column - Examples & Info */}
          <div className="space-y-6">
            {/* Try Examples */}
            <div className={`rounded-2xl border-2 p-6 overflow-hidden ${isDark ? cardDark : cardLight}`}>
              <h2 className={`text-lg font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Try Examples
              </h2>
              <div className="space-y-2">
                {exampleTexts.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => tryExample(example)}
                    disabled={loading}
                    className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors disabled:opacity-50 border ${
                      isDark
                        ? 'text-slate-300 hover:bg-fuchsia-900/30 hover:text-fuchsia-200 border-fuchsia-600'
                        : 'text-gray-700 hover:bg-orange-100 hover:text-orange-700 border-orange-300'
                    }`}
                  >
                    "{example.length > 60 ? example.slice(0, 60) + '...' : example}"
                  </button>
                ))}
              </div>
            </div>

            {/* Emotion Legend */}
            <div className={`rounded-2xl border-2 p-6 overflow-hidden ${isDark ? cardDark : cardLight}`}>
              <h2 className={`text-lg font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Emotions We Detect
              </h2>
              <div className="space-y-3">
                {Object.entries(emotionConfig).map(([emotion, config]) => (
                  <div key={emotion} className="flex items-center space-x-3">
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${config.gradient} flex items-center justify-center text-xl`}>
                      {config.emoji}
                    </div>
                    <div>
                      <p className={`font-medium capitalize ${isDark ? 'text-white' : 'text-gray-900'}`}>
                        {emotion}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Info Card */}
            <div className={`rounded-2xl border-2 p-6 text-white overflow-hidden ${isDark ? 'bg-gradient-to-br from-fuchsia-900 to-purple-900 border-fuchsia-600' : 'bg-gradient-to-br from-orange-500 to-red-600 border-orange-400'}`}>
              <h3 className="font-semibold mb-2">About MoodFlix</h3>
              <p className="text-sm text-white/90">
                AI-powered emotion detection meets cinema. Share your feelings and discover movies that match your mood. Powered by DeBERTa v3 and TMDB.
              </p>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

export default EmotionDetector;