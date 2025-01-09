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
          "bookid": "0"
        },
        {
          "name": "Lord of the Rings",
          "url": "https://m.media-amazon.com/images/I/913sMwNHsBL._SL1500_.jpg",
          "bookid": "1"
        },
        {
          "name": "The Perfume",
          "url": "https://upload.wikimedia.org/wikipedia/en/f/f5/PerfumeSuskind.jpg",
          "bookid": "2"
        },
        {
          "name": "Fall or Dodge; in Hell",
          "url": "https://m.media-amazon.com/images/I/91Z1I-2LQpL._AC_UF1000,1000_QL80_.jpg",
          "bookid": "3"
        },
        {
          "name": "Brave New World",
          "url": "https://m.media-amazon.com/images/I/91D4YvdC0dL._AC_UF894,1000_QL80_.jpg",
          "bookid": "4"
        },
        {
          "name": "The Midnight Library",
          "url": "https://media.s-bol.com/yyE0mpJQW856/o2jMxV3/547x840.jpg",
          "bookid": "5"
        },
        {
          "name": "Cryptonomicon",
          "url": "https://m.media-amazon.com/images/I/811HmmUbx3L._AC_UF1000,1000_QL80_.jpg",
          "bookid": "6"
        },
        {
          "name": "Hersenschimmen",
          "url": "https://media.s-bol.com/vZg12A3m36QM/739x1200.jpg",
          "bookid": "7"
        },
        {
          "name": "Tokyo Vice",
          "url": "https://m.media-amazon.com/images/I/61TWK--pVVL._AC_UF894,1000_QL80_.jpg",
          "bookid": "8"
        }
      ]
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
