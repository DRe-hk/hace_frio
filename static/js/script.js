// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    const carousel = document.getElementById('carousel');
    const currentHourCard = document.getElementById('current-hour-card');

    if (carousel && currentHourCard) {
        // Calcular la posición para centrar la tarjeta
        const scrollLeft = currentHourCard.offsetLeft - (carousel.offsetWidth / 2) + (currentHourCard.offsetWidth / 2);
        
        // Desplazar el carrusel a esa posición
        carousel.scrollLeft = scrollLeft;
    }
});
