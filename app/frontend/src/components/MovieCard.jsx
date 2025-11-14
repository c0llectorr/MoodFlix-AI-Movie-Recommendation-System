import React from 'react';
import { Star, Calendar } from 'lucide-react';

const MovieCard = ({ movie, isDark = false }) => {
  const posterUrl = movie.poster_path
    ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
    : 'https://via.placeholder.com/300x450?text=No+Image';

  const releaseYear = movie.release_date
    ? new Date(movie.release_date).getFullYear()
    : 'N/A';

  return (
    <div
      className={`flex-shrink-0 w-40 rounded-lg overflow-hidden shadow-lg transform transition-all duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer ${
        isDark
          ? 'bg-gray-800 border border-purple-600'
          : 'bg-white border border-yellow-300'
      }`}
    >
      {/* Poster Image */}
      <div className="relative overflow-hidden h-64 bg-gradient-to-br from-gray-300 to-gray-400">
        <img
          src={posterUrl}
          alt={movie.title}
          className="w-full h-full object-cover"
          loading="lazy"
        />
        {/* Rating Badge */}
        <div
          className={`absolute top-2 right-2 flex items-center space-x-1 px-2 py-1 rounded-full backdrop-blur-sm ${
            isDark
              ? 'bg-purple-600/90'
              : 'bg-yellow-400/90'
          }`}
        >
          <Star className="w-3 h-3 fill-current" />
          <span className={`text-xs font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            {movie.vote_average.toFixed(1)}
          </span>
        </div>
      </div>

      {/* Card Content */}
      <div className={`p-3 ${isDark ? 'bg-gray-900' : 'bg-white'}`}>
        {/* Title */}
        <h3
          className={`text-sm font-bold line-clamp-2 mb-2 ${
            isDark ? 'text-white' : 'text-gray-900'
          }`}
          title={movie.title}
        >
          {movie.title}
        </h3>

        {/* Release Year */}
        <div className="flex items-center space-x-1">
          <Calendar className={`w-3 h-3 ${isDark ? 'text-purple-400' : 'text-yellow-600'}`} />
          <span
            className={`text-xs ${
              isDark ? 'text-gray-400' : 'text-gray-600'
            }`}
          >
            {releaseYear}
          </span>
        </div>
      </div>
    </div>
  );
};

export default MovieCard;
