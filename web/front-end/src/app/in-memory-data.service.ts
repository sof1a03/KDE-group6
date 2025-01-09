import { Injectable } from '@angular/core';
import { InMemoryDbService } from 'angular-in-memory-web-api';

@Injectable({
  providedIn: 'root',
})
export class InMemoryDataService implements InMemoryDbService {
  createDb() {
    const books = [
      {
        "name": "Flowers for Algernon",
        "url": "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg",
        "bookid": "0",
        "genres": ["Science Fiction", "Drama"],
        "publisher": "Harcourt, Brace & World",
        "year": 1966,
        "ISBN": "978-0-15-632940-9"
      },
      {
        "name": "Lord of the Rings",
        "url": "https://m.media-amazon.com/images/I/913sMwNHsBL._SL1500_.jpg",
        "bookid": "1",
        "genres": ["Fantasy", "Adventure"],
        "publisher": "George Allen & Unwin",
        "year": 1954,
        "ISBN": "978-0-618-00221-3"
      },
      {
        "name": "The Perfume",
        "url": "https://upload.wikimedia.org/wikipedia/en/f/f5/PerfumeSuskind.jpg",
        "bookid": "2",
        "genres": ["Historical Fiction", "Thriller"],
        "publisher": "Diogenes Verlag",
        "year": 1985,
        "ISBN": "978-0-394-53982-5"
      },
      {
        "name": "Fall or Dodge; in Hell",
        "url": "https://m.media-amazon.com/images/I/91Z1I-2LQpL._AC_UF1000,1000_QL80_.jpg",
        "bookid": "3",
        "genres": ["Science Fiction", "Satire"],
        "publisher": "Simon & Schuster",
        "year": 2019,
        "ISBN": "978-1-4767-3802-9"
      },
      {
        "name": "Brave New World",
        "url": "https://m.media-amazon.com/images/I/91D4YvdC0dL._AC_UF894,1000_QL80_.jpg",
        "bookid": "4",
        "genres": ["Science Fiction", "Dystopian"],
        "publisher": "Chatto & Windus",
        "year": 1932,
        "ISBN": "978-0-06-085052-4"
      },
      {
        "name": "The Midnight Library",
        "url": "https://media.s-bol.com/yyE0mpJQW856/o2jMxV3/547x840.jpg",
        "bookid": "5",
        "genres": ["Fantasy", "Magical Realism"],
        "publisher": "Canongate Books",
        "year": 2020,
        "ISBN": "978-1-78689-273-5"
      },
      {
        "name": "Cryptonomicon",
        "url": "https://m.media-amazon.com/images/I/811HmmUbx3L._AC_UF1000,1000_QL80_.jpg",
        "bookid": "6",
        "genres": ["Historical Fiction", "Science Fiction"],
        "publisher": "Spectra Books",
        "year": 1999,
        "ISBN": "978-0-380-78862-0"
      },
      {
        "name": "Hersenschimmen",
        "url": "https://media.s-bol.com/vZg12A3m36QM/739x1200.jpg",
        "bookid": "7",
        "genres": ["Psychological Fiction", "Drama"],
        "publisher": "De Bezige Bij",
        "year": 1984,
        "ISBN": "978-90-234-5363-9"
      },
      {
        "name": "Tokyo Vice",
        "url": "https://m.media-amazon.com/images/I/61TWK--pVVL._AC_UF894,1000_QL80_.jpg",
        "bookid": "8",
        "genres": ["True Crime", "Memoir"],
        "publisher": "Pantheon Books",
        "year": 2009,
        "ISBN": "978-0-307-37879-0"
      }
    ];
    return {books};
    }

  // Overrides the genId method to ensure that a hero always has an id.
  // If the heroes array is empty,
  // the method below returns the initial number (11).
  // if the heroes array is not empty, the method below returns the highest
  // hero id + 1.
 // genId(heroes: Hero[]): number {
  //  return heroes.length > 0 ? Math.max(...heroes.map(hero => hero.id)) + 1 : 11;
  //}
}
