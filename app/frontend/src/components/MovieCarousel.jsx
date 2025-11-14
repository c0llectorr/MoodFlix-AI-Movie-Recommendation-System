import React, { useRef } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import MovieCard from './MovieCard';

const MovieCarousel = ({ genre, movies, isDark = false }) => {
  const scrollContainerRef = useRef(null);

  const scroll = (direction) => {
    if (scrollContainerRef.current) {
      const scrollAmount = 400;
      if (direction === 'left') {
        scrollContainerRef.current.scrollBy({ left: -scrollAmount, behavior: 'smooth' });
      } else {
        scrollContainerRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
      }
    }
  };

  return (
    <div
      className={`rounded-xl overflow-hidden ${
        isDark
          ? 'bg-gradient-to-r from-gray-900 via-purple-900 to-gray-900 border border-purple-700'
          : 'bg-gradient-to-r from-yellow-50 via-green-50 to-yellow-50 border border-green-300'
      }`}
    >
      {/* Genre Header */}
      <div className={`px-6 py-4 flex items-center justify-between ${isDark ? 'bg-gray-800/50' : 'bg-green-100/50'}`}>
        <h2
          className={`text-2xl font-bold capitalize ${
            isDark
              ? 'text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400'
              : 'text-transparent bg-clip-text bg-gradient-to-r from-green-700 to-yellow-700'
          }`}
        >
          {genre}
        </h2>
        <span
          className={`text-sm font-semibold px-3 py-1 rounded-full ${
            isDark ? 'bg-purple-600 text-white' : 'bg-yellow-300 text-gray-900'
          }`}
        >
          {movies.length} movies
        </span>
      </div>

      {/* Carousel Container */}
      <div className="relative group">
        {/* Left Arrow */}
        <button
          onClick={() => scroll('left')}
          className={`absolute left-2 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full opacity-0 group-hover:opacity-100 transform transition-all duration-300 ${
            isDark
              ? 'bg-purple-600 hover:bg-purple-700 text-white'
              : 'bg-yellow-400 hover:bg-yellow-500 text-gray-900'
          }`}
        >
          <ChevronLeft className="w-5 h-5" />
        </button>

        {/* Scrollable Container */}
        <div
          ref={scrollContainerRef}
          className="flex gap-4 overflow-x-auto scroll-smooth px-6 py-4 scrollbar-hide"
          style={{ scrollBehavior: 'smooth', scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {movies.map((movie) => (
            <MovieCard key={movie.id} movie={movie} isDark={isDark} />
          ))}
        </div>

        {/* Right Arrow */}
        <button
          onClick={() => scroll('right')}
          className={`absolute right-2 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full opacity-0 group-hover:opacity-100 transform transition-all duration-300 ${
            isDark
              ? 'bg-purple-600 hover:bg-purple-700 text-white'
              : 'bg-yellow-400 hover:bg-yellow-500 text-gray-900'
          }`}
        >
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>

      {/* Custom Scrollbar Hide */}
      <style jsx>{`
        div::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </div>
  );
};

export default MovieCarousel;
