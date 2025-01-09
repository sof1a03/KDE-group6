import { Routes } from '@angular/router';
import { BookViewComponent } from './components/book-view/book-view.component';
import { BookPageComponent } from './components/book-page/book-page.component';

export const routes: Routes = [
{path: 'featured', component: BookViewComponent
},
{
  path: 'book/:id', component: BookPageComponent
}
];
